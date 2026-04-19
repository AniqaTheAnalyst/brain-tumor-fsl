# 2_models_and_features.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import os
import shutil
import open_clip

# Import from File 1
from 1_data_and_setup import (
    DEVICE, IMAGE_SIZE, support_pool, query_pool, 
    N_WAY, CLASS_NAMES, cache_dir
)

# ========================= MODELS =========================
print("Loading models...")
biomed_model, biomed_preprocess = open_clip.create_model_from_pretrained(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
biomed_visual = biomed_model.visual.to(DEVICE)
biomed_visual.eval()

resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()
resnet = resnet.to(DEVICE)
resnet.eval()

efficientnet = models.efficientnet_b0(pretrained=True)
efficientnet.classifier = nn.Identity()
efficientnet = efficientnet.to(DEVICE)
efficientnet.eval()

imagenet_preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model_configs = {
    "BioMedCLIP": {"encoder": biomed_visual, "preprocess": biomed_preprocess, "dim": 768},
    "ResNet18": {"encoder": resnet, "preprocess": imagenet_preprocess, "dim": 512},
    "EfficientNet-B0": {"encoder": efficientnet, "preprocess": imagenet_preprocess, "dim": 1280}
}
print("✅ Models loaded.\n")

# ========================= FEATURE CACHING =========================
cache_dir = "/kaggle/working/feature_cache_pub"
if os.path.exists(cache_dir):
    print("⚠️ Clearing old feature cache...")
    shutil.rmtree(cache_dir)
os.makedirs(cache_dir, exist_ok=True)

def extract_features(name, encoder, preprocess, classwise_dict, tag):
    cache_file = os.path.join(cache_dir, f"{name}_{tag}.pt")
    if os.path.exists(cache_file):
        print(f" ✓ Cached [{tag}] {name}")
        return torch.load(cache_file, map_location='cpu')
    
    all_paths = [p for paths in classwise_dict.values() for p in paths]
    features = {}
    with torch.no_grad():
        for path in tqdm(all_paths, desc=f"{name}[{tag}]"):
            try:
                img = Image.open(path).convert("RGB")
                tensor = preprocess(img).unsqueeze(0).to(DEVICE)
                feat = encoder(tensor).squeeze(0).cpu()
                features[path] = feat
            except Exception as e:
                print(f" ⚠ Skipped {path}: {e}")
    torch.save(features, cache_file)
    return features

print("Pre-extracting features...")
support_feats = {}
query_feats = {}
for name, cfg in model_configs.items():
    support_feats[name] = extract_features(name, cfg["encoder"], cfg["preprocess"], support_pool, "train")
    query_feats[name] = extract_features(name, cfg["encoder"], cfg["preprocess"], query_pool, "test")
print("✅ Feature extraction done.\n")

print("✅ Models and features ready.")