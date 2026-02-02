import argparse
import os
import io
import csv
import datetime
import numpy as np
import random
from typing import Tuple, Sequence, Optional
from PIL import Image, ImageFile
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from sklearn.metrics import roc_auc_score, average_precision_score

# Local imports
from models.mirror import build_mirror



# === 0. Setup Environment & Seeds ===
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Suppress deterministic warnings
ImageFile.LOAD_TRUNCATED_IMAGES = True

def seed_everything(seed: int = 42, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"[INFO] Random seed set to {seed}, deterministic={deterministic}")

seed_everything(0, deterministic=True)

# ==========================================
# 1. Image Augmentation & Transform Utils
#    (Imported from training code)
# ==========================================

try:
    BICUBIC = InterpolationMode.BICUBIC
except Exception:
    BICUBIC = Image.BICUBIC

def compress_image(img_pil, quality=96):
    """Simulate JPEG compression artifacts in memory for PNG inputs."""
    img = img_pil.convert('RGB')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality, optimize=True)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')

class RandomScaleCropOrDirect224:
    """
    Advanced cropping strategy.
    Eval modes: 'short256_center', 'direct'
    """
    def __init__(
        self,
        crop_size: int = 224,
        interpolation: InterpolationMode = BICUBIC,
        antialias: bool = True,
        infer_policy: str = "short256_center",
        eval_resize_short: int = 256,
    ):
        self.crop_size = crop_size
        self.interp = interpolation
        self.antialias = antialias
        self.infer_policy = infer_policy
        self.eval_resize_short = eval_resize_short

    def _get_size(self, img):
        return img.size if isinstance(img, Image.Image) else (img.shape[2], img.shape[1])

    def _resize_short(self, img, target_short: int):
        w, h = self._get_size(img)
        short = min(w, h)
        if short == target_short: return img
        scale = target_short / short
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        return TF.resize(img, [new_h, new_w], interpolation=self.interp, antialias=self.antialias)

    def __call__(self, img):
        # Default: Resize short side to eval_resize_short then center crop
        w, h = self._get_size(img)
        if min(w, h) > self.eval_resize_short or self.infer_policy == "resize256_center":
             img = self._resize_short(img, self.eval_resize_short)
        return TF.center_crop(img, [self.crop_size, self.crop_size])

def Get_Transforms(args, mode='preprocess'):
    if mode == 'preprocess':
        # Removed train_t logic as we only need eval here
        eval_t = [
            RandomScaleCropOrDirect224(
                infer_policy="short256_center",
                eval_resize_short=512 # Ensuring high res input for validation
            ),
            transforms.CenterCrop(224),
        ]
        return None, transforms.Compose(eval_t)
        
    elif mode == 'normal' or mode == 'fake':
        # Standard normalization for model input
        eval_t = [transforms.ToTensor()]
        return None, transforms.Compose(eval_t)
    
    return None, None

# ==========================================
# 2. Datasets (Imported from training code)
# ==========================================

class TestDataset(Dataset):
    def __init__(self, args):
        self.root = args.eval_data_path
        self.data_list = []
        
        # Get Eval transforms
        _, self.transform_pre = Get_Transforms(args, 'preprocess')
        _, self.transform_norm = Get_Transforms(args, 'normal')
        
        # Quietly load to avoid spamming logs in loops
        # print(f"Loading Test Data from: {self.root}") 
        for root, _, files in os.walk(self.root):
            label = None
            if os.path.basename(root) == "0_real": label = 0
            elif os.path.basename(root) == "1_fake": label = 1
            
            if label is not None:
                for img in files:
                    if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        self.data_list.append({"path": os.path.join(root, img), "label": label})

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        try:
            sample = self.data_list[index]
            path, target = sample['path'], sample['label']
            
            image = Image.open(path).convert('RGB')
            image = self.transform_pre(image)
            
            # Simulate JPEG alignment if testing on PNGs (common in forensic benchmarks)
            if path.lower().endswith(('.png', '.bmp', '.tiff', '.PNG')):
                 image = compress_image(image, quality=96)
            
            image = self.transform_norm(image)
            return torch.tensor(image), torch.tensor(int(target))
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

# ============================
# 3. Configuration & Paths
# ============================

DATASET_CONFIGS = {
    'AIGCDetect': ["sd_xl", "progan", "stylegan", "biggan", "cyclegan", "stargan", "gaugan", "stylegan2", "whichfaceisreal", "ADM", "Glide", "Midjourney", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "VQDM", "wukong", "DALLE2"],
    'Genimage': ["Midjourney", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "ADM", "Glide", "wukong", "VQDM", "biggan2"],
    'ojha': ["deepfake", "san", "crn", "imle", "guided", "ldm_100", "ldm_200", "ldm_200_cfg", "glide_50_27", "glide_100_10", "glide_100_27", "dalle"],
    'WildRF': ["facebook", "reddit", "twitter"],
    'Synthbuster': ["dalle2", "dalle3", "firefly", "glide", "midjourney-v5", "stable-diffusion-1-3", "stable-diffusion-1-4", "stable-diffusion-2", "stable-diffusion-xl"],
    'Synthwildx': ["dalle3", "firefly", "midjourney_v5"],
    'DRCT': ["real", "SDXL-DR", "LDM", "SDv1.4", "SDv1.5", "SDv2", "SDXL", "SDXL-Refiner", "SD-Turbo", "SDXL-Turbo", "LCM-SDv1.5", "LCM-SDXL", "SDv1-Ctrl", "SDv2-Ctrl", "SDXL-Ctrl", "SDv1-DR", "SDv2-DR"],
    'AIGIBench': ["SocialRF", "CommunityAI"],
    'UnivFD': ['deepfake', 'seeingdark', 'san', 'ldm_200_cfg', 'ldm_200', 'ldm_100', 'imle', 'guided', 'glide_50_27', 'glide_100_27', 'glide_100_10', 'dalle', 'crn'],
    'EvalGEN': ['OmniGen', 'NOVA', 'Infinity', 'GoT', 'Flux'],
    'CO-SPY': ['midjourney', 'lexica', 'flux', 'dalle3', 'civitai'],
    'realchain_all': ['flux', 'hunyuan3.0', 'nanobanana', 'qwenimage', 'sd3.5', 'seedream_i2i', 'seedream4'],
    'AIGI-Human-orig': ['SD-3-Medium', 'SD-3.5-Large', 'FLUX.1-dev', 'PixArt-Sigma', 'Midjourney-v6', 'DALL-E-3'],
    'AIGI-Human-hard': ['SD-3-Medium', 'SD-3.5-Large', 'FLUX.1-dev', 'PixArt-Sigma', 'Midjourney-v6', 'DALL-E-3']
}
DATASET_CONFIGS['realchain_CD_all'] = DATASET_CONFIGS['realchain_all']

BENCHMARK_PATHS = {
    'AIGI-Human-hard': 'aigi_human',
    'AIGI-Human-orig': 'human_aigi_orig',
    'AIGI-Now': 'AIGI-Now',
    'AIGCDetect': 'AIGC_bm',
    'Synthwildx': 'synthwildx',
    'WildRF': 'WildRF/test',
    'Chameleon': 'Chameleon/test',
    'realchain_all': 'realchain_all',
    'realchain_CD_all': 'realchain_CD_all',
    'AIGIBench': 'AIGIBench',
    'DRCT': 'drct',
    'CO-SPY': 'CO-SPY-In-the-Wild',
    'RRDataset': 'RRDataset',
    'B-Free': 'B-Free',
    'EvalGEN': 'GenEval-JPEG',
    'Synthbuster': 'synthbuster',
    'UnivFD': 'UniversalFakeDetect'
}

# ============================
# 4. Evaluation Function
# ============================

def evaluate(data_loader, model, device, use_amp=False):
    model.eval()
    all_preds, all_labels, all_scores = [], [], []
    
    with torch.no_grad():
        for samples, targets in tqdm(data_loader, desc="   Computing", leave=False):
            samples = samples.to(device, non_blocking=True)
            
            # Updated autocast syntax to fix FutureWarning
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, _, _ = model(samples)
                
            all_scores.extend(torch.nn.functional.softmax(logits, dim=1)[:, 1].cpu().numpy())
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(targets.numpy())

    all_preds, all_labels, all_scores = np.array(all_preds), np.array(all_labels), np.array(all_scores)
    
    if len(all_labels) == 0: return 0, 0, 0, 0, 0, 0
    
    acc = np.mean(all_preds == all_labels)
    acc_real = np.mean(all_preds[all_labels == 0] == 0) if np.any(all_labels == 0) else 0.0
    acc_fake = np.mean(all_preds[all_labels == 1] == 1) if np.any(all_labels == 1) else 0.0
    bal_acc = (acc_real + acc_fake) / 2
    
    try:
        auc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.5
        ap = average_precision_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0
    except:
        auc, ap = 0.5, 0.0
        
    return acc, acc_real, acc_fake, bal_acc, auc, ap

# ============================
# 5. Main Execution
# ============================

def get_args():
    parser = argparse.ArgumentParser('MIRROR Multi-Benchmark Inference')
    parser.add_argument('--model_path', default="/data/liurq_data/outputs/mirror/mirror_phase2/checkpoint-h-cur.pth", type=str)
    parser.add_argument('--memory_path', default='/data/liurq_data/outputs/mirror/mirror_phase1/mirror_phase1_epoch_2.pth', type=str)
    parser.add_argument('--backbone_path', default='/data/liurq_data/weights/dinov3-huge', type=str)
    parser.add_argument('--base_data_path', default='/data/liurq_data/sid_data', type=str)
    parser.add_argument('--benchmarks', nargs='+', default=['Chameleon'], help='List of benchmarks')
    parser.add_argument('--output_dir', default='./results', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--use_amp', action='store_true')
    
    # Placeholder for arg used in TestDataset
    parser.add_argument('--eval_data_path', default=None, type=str, help='Internal use only')
    return parser.parse_args()

def main(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. Load Model
    print(f">>> Loading Model: {os.path.basename(args.model_path)}")
    model = build_mirror(memory_path=args.memory_path, backbone_path=args.backbone_path)
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint.get('model', checkpoint.get('state_dict', checkpoint)), strict=False)
    model.to(device)

    # 2. Iterate Benchmarks
    for benchmark in args.benchmarks:
        folder_name = BENCHMARK_PATHS.get(benchmark, benchmark)
        root_dir = os.path.join(args.base_data_path, folder_name)
        sub_datasets = DATASET_CONFIGS.get(benchmark, [None]) 
        
        print(f"\n[{benchmark}] Starting... (Root: {folder_name})")
        
        benchmark_results = [] 
        benchmark_metrics = []

        # 3. Iterate Sub-datasets
        for sub in sub_datasets:
            dataset_path = os.path.join(root_dir, sub) if sub else root_dir
            display_name = sub if sub else 'ALL'
            
            if not os.path.exists(dataset_path):
                print(f"  [Skip] Path not found: {dataset_path}")
                continue
            
            # --- CRITICAL CHANGE: Use TestDataset from training code ---
            # Update args.eval_data_path so TestDataset picks the right folder
            args.eval_data_path = dataset_path 
            dataset = TestDataset(args) 

            if len(dataset) == 0:
                print(f"  [Skip] No images in: {display_name}")
                continue

            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                                sampler=SequentialSampler(dataset), pin_memory=True)
            
            print(f"  Evaluating {display_name:<25} ...", end='\r')
            acc, acc_real, acc_fake, bal_acc, auc, ap = evaluate(loader, model, device, args.use_amp)
            print(f"  {display_name:<25} | Bal_Acc: {bal_acc:.2%} | AUC: {auc:.4f}")
            
            benchmark_results.append([display_name, acc, acc_real, acc_fake, bal_acc, auc, ap])
            benchmark_metrics.append([acc, acc_real, acc_fake, bal_acc, auc, ap])

        # 4. Calculate Mean
        if len(benchmark_metrics) > 1:
            mean_vals = np.mean(np.array(benchmark_metrics), axis=0)
            benchmark_results.append(['MEAN'] + mean_vals.tolist())
            print(f"  >>> {benchmark} MEAN      | Bal_Acc: {mean_vals[3]:.2%} | AUC: {mean_vals[4]:.4f}")

        # 5. Save Individual CSV
        if benchmark_results:
            csv_filename = f"{benchmark}_{timestamp}.csv"
            csv_path = os.path.join(args.output_dir, csv_filename)
            headers = ['Dataset', 'Acc', 'Real_Acc', 'Fake_Acc', 'Bal_Acc', 'AUC', 'AP']
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(benchmark_results)
            print(f"  [Saved] Results for {benchmark} -> {csv_filename}")
        else:
            print(f"  [Warn] No results to save for {benchmark}")

    print("\nAll tasks completed.")

if __name__ == '__main__':
    args = get_args()
    main(args)