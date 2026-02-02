import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import get_peft_model, LoraConfig

# ==========================================
# 1. Normalization & Backbone
# ==========================================
class Norm(nn.Module):
    def __init__(self, mode='clip'):
        super().__init__()
        self.mode = mode
        if mode == 'clip':
            self.register_buffer('mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        else:  # imagenet (Standard for DINO)
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

class DINO(nn.Module):
    def __init__(self, dinov3_path, finetune=True):
        super(DINO, self).__init__()
        print(f"Loading Backbone from: {dinov3_path}")
        self.dino = AutoModel.from_pretrained(dinov3_path, weights_only=False)
        self.dino.requires_grad_(False)
        
        # Identify model type for LoRA targeting
        # This handles both DINOv2 (facebook) and DINOv3 architectures
        self.is_v3 = hasattr(self.dino, "layer") or "dinov3" in dinov3_path.lower()
        
        if finetune:
            self._apply_lora()

    def _apply_lora(self):
        # Target QKV for LoRA as per standard practice
        target_modules = ["q_proj", "k_proj", "v_proj"] if self.is_v3 else ["query", "key", "value"]
        config = LoraConfig(r=8, lora_alpha=16, target_modules=target_modules)
        
        # Apply LoRA to encoder layers
        encoder_layers = self.dino.layer if self.is_v3 else self.dino.encoder.layer
        for i in range(len(encoder_layers)):
            encoder_layers[i] = get_peft_model(encoder_layers[i], config)

    def forward(self, x):
        outputs = self.dino(pixel_values=x)
        last_hidden_state = outputs[0]
        
        # Separate CLS token and Patch tokens
        # Assuming index 0 is CLS (Standard ViT)
        feat_cls = last_hidden_state[:, 0]
        feat_tokens = last_hidden_state[:, 1:]
        return feat_tokens, feat_cls

# ==========================================
# 2. Memory Bank (Phase 1 Frozen)
# ==========================================
class MirrorMemoryBank(nn.Module):
    """
    Implements the Memory Bank with Top-k Sparse Projection.
    Aligned with Phase 1 logic.
    """
    def __init__(self, feature_dim, mem_slots=4096, num_heads=8, top_k=128):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.top_k = top_k

        # Memory matrix (Loaded from Phase 1)
        self.memory = nn.Parameter(torch.randn(mem_slots, feature_dim))
        
        # Projections
        self.q_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.k_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.v_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        B, N, C = x.shape
        M_slots, _ = self.memory.shape
        
        # 1. Projections
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Expand memory for batch processing
        mem_ext = self.memory.unsqueeze(0).expand(B, -1, -1)
        k = self.k_proj(mem_ext).reshape(B, M_slots, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(mem_ext).reshape(B, M_slots, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Calculate Attention Logits
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale # [B, H, N, M]

        # 3. Top-k Sparsity (Crucial for MIRROR)
        # We only want the top-k most relevant prototypes to reconstruct the reference
        topk_vals, topk_indices = torch.topk(attn_logits, k=self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_vals, dim=-1) # [B, H, N, k]

        # 4. Reconstruction
        # Gather the corresponding V values
        # v: [B, H, M, D_head] -> select topk -> [B, H, N, k, D_head]
        # This gather can be expensive; optimized implementation might assume flat indexing
        # For clarity here, we simulate the selection:
        
        # Note: A full gather is complex to write efficiently in pure PyTorch without einsum/indexing tricks
        # A simplified approximation if we assume we just weight the top k values:
        
        # Zero out non-top-k values in the full attention matrix for reconstruction
        mask = torch.zeros_like(attn_logits).scatter_(-1, topk_indices, 1.0)
        sparse_attn = torch.softmax(attn_logits.masked_fill(mask == 0, -1e9), dim=-1)
        
        # 5. Reconstruct Ideal Reference
        recon = (sparse_attn @ v).transpose(1, 2).reshape(B, N, C)
        recon = self.out_proj(recon)

        # Return full logits for Perplexity calculation (Entropy needs distribution)
        # Or return sparse_attn depending on how entropy is defined in your exp.
        # Paper implies entropy of the retrieval, typically on the sparse or full distribution.
        return recon, sparse_attn

# ==========================================
# 3. Detection Head (Phase 2)
# ==========================================
class DualBranchClassifier(nn.Module):
    def __init__(self, feat_dim, hidden_dim=512):
        super().__init__()
        
        # Branch 1: Reconstruct Perplexity (Uncertainty)
        # Inputs: Max Attention Score & Entropy
        self.perplexity_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Branch 2: Comparison Residual (Detail Deviation)
        # Input: Difference vector (F - F_hat)
        self.residual_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 256)
        )
        
        # Final Classification Head
        # Concatenates Evidence vectors V_per and V_res
        self.head = nn.Sequential(
            nn.Linear(64 + 256 + feat_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, attn_weights, f_last, f_recon, feat_cls):
        # 1. Calculate Perplexity Metrics (Eq. 4)
        # attn_weights: [B, H, N, M]
        # Average over Heads and Patches to get global image-level uncertainty? 
        # Or per-patch? Paper suggests global signals for the final classifier.
        
        # Max Score: Confidence of finding a prototype
        max_scores = attn_weights.max(dim=-1)[0].mean(dim=[1, 2]).unsqueeze(-1) # [B, 1]
        
        # Entropy: Uncertainty in prototype selection
        # Add epsilon to avoid log(0)
        entropy = -(attn_weights * torch.log(attn_weights + 1e-9)).sum(dim=-1).mean(dim=[1, 2]).unsqueeze(-1) # [B, 1]
        
        # V_per
        v_per = self.perplexity_mlp(torch.cat([max_scores, entropy], dim=-1)) # [B, 64]

        # 2. Calculate Residual Metrics
        # Residual = F - F_hat
        residual_map = f_last - f_recon
        # Global Average Pooling of residuals to get image-level deviation vector
        residual_gap = residual_map.mean(dim=1) # [B, D]
        
        # V_res
        v_res = self.residual_mlp(residual_gap) # [B, 256]

        # 3. Final Prediction
        # Strictly Reference-Comparison: Only use V_per and V_res
        combined = torch.cat([v_per, v_res, feat_cls], dim=-1)
        
        return self.head(combined)

# ==========================================
# 4. Full Model Assembly (MIRROR)
# ==========================================
class MIRROR_Detector(nn.Module):
    def __init__(self, dino_path, memory_path=None, feature_dim=1024):
        super(MIRROR_Detector, self).__init__()
        
        # 1. Preprocessing
        self.norm = Norm(mode='imagenet')
        
        # 2. Backbone (DINOv3 with LoRA)
        self.backbone = DINO(dino_path, finetune=True)
        
        # 3. Memory Bank (Frozen Reality Priors)
        # Ensure dimensions match Phase 1 settings
        self.memory_bank = MirrorMemoryBank(feature_dim=feature_dim)
        
        if memory_path:
            print(f"Loading Memory Bank from {memory_path}")
            state_dict = torch.load(memory_path, map_location='cpu')
            # Handle potential key mismatches if saved with 'module.' prefix
            clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict['model_state_dict'].items()}
            self.memory_bank.load_state_dict(clean_state_dict, strict=False)
        
        # Freeze Memory Bank
        self.memory_bank.eval()
        for param in self.memory_bank.parameters():
            param.requires_grad = False
            
        # 4. Detector Head
        self.detector = DualBranchClassifier(feat_dim=feature_dim)

    def forward(self, x):
        # Norm
        x = self.norm(x)
        
        # Extract Features (LoRA active)
        feat_tokens, feat_cls = self.backbone(x)
        
        # Memory Retrieval (Frozen)
        with torch.no_grad():
            f_recon, attn_weights = self.memory_bank(feat_tokens)
            
        # Detection (Reference-Comparison)
        logits = self.detector(attn_weights, feat_tokens, f_recon, feat_cls)
        
        return logits, f_recon, feat_tokens

def build_mirror(memory_path='', backbone_path=''):
    # Determine dimension based on backbone name or config
    if 'large' in backbone_path:
        dim = 1024
    elif 'huge' in backbone_path:
        dim = 1280
    else:
        dim = 768 # Base/Small
        
    model = MIRROR_Detector(dino_path=backbone_path, memory_path=memory_path, feature_dim=dim)
    return model

# Example usage
if __name__ == "__main__":
    # Simulate loading
    # model = build_mirror(memory_path='./checkpoints/mirror_phase1/mirror_phase1_epoch_5.pth')
    pass