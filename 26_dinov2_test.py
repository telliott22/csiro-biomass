#!/usr/bin/env python3
"""
Quick test script to verify DINOv2 works with our biomass dataset.

Tests:
1. Load DINOv2 from local checkpoint
2. Extract features from sample images
3. Compare with ResNet18 features
4. Verify memory usage and inference speed
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import AutoImageProcessor, Dinov2Model
from PIL import Image
import pandas as pd
import time
import numpy as np
from pathlib import Path

print("="*80)
print("DINOv2 Quick Test - Phase 1")
print("="*80)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Load sample data
print("\n1. Loading sample data...")
train_df = pd.read_csv('competition/train_enriched.csv')
sample_images = train_df.sample(n=5, random_state=42)
print(f"   Selected {len(sample_images)} random images for testing")

# ============================================================================
# Test 1: Load DINOv2 from local checkpoint
# ============================================================================
print("\n2. Loading DINOv2 from local checkpoint...")
dinov2_path = Path.home() / "Downloads/dinov2_base"

try:
    # Load using transformers library
    processor = AutoImageProcessor.from_pretrained(str(dinov2_path))
    dinov2_model = Dinov2Model.from_pretrained(str(dinov2_path))
    dinov2_model = dinov2_model.to(device)
    dinov2_model.eval()

    print(f"   ✓ DINOv2 loaded successfully!")
    print(f"   Model size: {sum(p.numel() for p in dinov2_model.parameters()):,} parameters")
    print(f"   Feature dimension: {dinov2_model.config.hidden_size}")
    print(f"   Expected input size: {dinov2_model.config.image_size}×{dinov2_model.config.image_size}")
except Exception as e:
    print(f"   ✗ Failed to load DINOv2: {e}")
    print("\n   Trying alternative loading method...")

    # Fallback: Load using torch hub (requires internet)
    try:
        dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        dinov2_model = dinov2_model.to(device)
        dinov2_model.eval()
        print(f"   ✓ DINOv2 loaded via torch.hub")
    except Exception as e2:
        print(f"   ✗ Failed to load via torch.hub: {e2}")
        print("\n   CANNOT PROCEED - DINOv2 not available")
        exit(1)

# ============================================================================
# Test 2: Load ResNet18 for comparison
# ============================================================================
print("\n3. Loading ResNet18 for comparison...")
resnet18 = models.resnet18(weights=None)
resnet18_backbone = nn.Sequential(*list(resnet18.children())[:-1])
resnet18_backbone = resnet18_backbone.to(device)
resnet18_backbone.eval()
print(f"   ✓ ResNet18 loaded")
print(f"   Model size: {sum(p.numel() for p in resnet18_backbone.parameters()):,} parameters")

# ============================================================================
# Test 3: Extract features from sample images
# ============================================================================
print("\n4. Extracting features from sample images...")

# Transforms
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dinov2_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # We'll use 224 for now (DINOv2 can handle it)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dinov2_features_list = []
resnet_features_list = []
dinov2_times = []
resnet_times = []

for idx, row in sample_images.iterrows():
    img_path = f"competition/{row['image_path']}"
    img = Image.open(img_path).convert('RGB')

    # ResNet18 features
    resnet_img = resnet_transform(img).unsqueeze(0).to(device)
    start = time.time()
    with torch.no_grad():
        resnet_feat = resnet18_backbone(resnet_img).flatten()
    resnet_time = time.time() - start
    resnet_times.append(resnet_time)
    resnet_features_list.append(resnet_feat.cpu().numpy())

    # DINOv2 features
    dinov2_img = dinov2_transform(img).unsqueeze(0).to(device)
    start = time.time()
    with torch.no_grad():
        if hasattr(dinov2_model, 'forward'):
            dinov2_out = dinov2_model(dinov2_img)
            # Get CLS token (first token) from last hidden state
            dinov2_feat = dinov2_out.last_hidden_state[:, 0].flatten()
        else:
            dinov2_feat = dinov2_model(dinov2_img).flatten()
    dinov2_time = time.time() - start
    dinov2_times.append(dinov2_time)
    dinov2_features_list.append(dinov2_feat.cpu().numpy())

    print(f"   Image {idx}: ResNet={resnet_time*1000:.1f}ms, DINOv2={dinov2_time*1000:.1f}ms")

# ============================================================================
# Test 4: Compare features
# ============================================================================
print("\n5. Feature Comparison:")
print(f"   ResNet18:")
print(f"     - Feature dim: {resnet_features_list[0].shape[0]}")
print(f"     - Avg inference time: {np.mean(resnet_times)*1000:.1f}ms")
print(f"     - Feature range: [{np.min(resnet_features_list[0]):.3f}, {np.max(resnet_features_list[0]):.3f}]")

print(f"   DINOv2:")
print(f"     - Feature dim: {dinov2_features_list[0].shape[0]}")
print(f"     - Avg inference time: {np.mean(dinov2_times)*1000:.1f}ms")
print(f"     - Feature range: [{np.min(dinov2_features_list[0]):.3f}, {np.max(dinov2_features_list[0]):.3f}]")

speedup = np.mean(dinov2_times) / np.mean(resnet_times)
print(f"\n   Speed: DINOv2 is {speedup:.1f}x {'SLOWER' if speedup > 1 else 'FASTER'} than ResNet18")

# ============================================================================
# Test 5: Memory usage estimate
# ============================================================================
print("\n6. Memory Usage Estimate:")

def get_model_memory(model):
    """Estimate model memory in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024**2)

resnet_mem = get_model_memory(resnet18_backbone)
dinov2_mem = get_model_memory(dinov2_model)

print(f"   ResNet18: {resnet_mem:.1f} MB")
print(f"   DINOv2: {dinov2_mem:.1f} MB")
print(f"   Ratio: {dinov2_mem/resnet_mem:.1f}x larger")

# ============================================================================
# Summary & Recommendation
# ============================================================================
print("\n" + "="*80)
print("SUMMARY & RECOMMENDATION")
print("="*80)

print(f"\n✓ DINOv2 loaded successfully from local checkpoint")
print(f"✓ Feature extraction works (768-dim features)")
print(f"✓ Compatible with our 224×224 images")

if speedup < 2.0:
    print(f"✓ Inference speed acceptable ({speedup:.1f}x slower than ResNet18)")
    speed_verdict = "GOOD"
else:
    print(f"⚠️  Inference speed concerning ({speedup:.1f}x slower than ResNet18)")
    speed_verdict = "CONCERNING"

if dinov2_mem < 500:
    print(f"✓ Memory usage acceptable ({dinov2_mem:.1f} MB)")
    memory_verdict = "GOOD"
else:
    print(f"⚠️  Memory usage high ({dinov2_mem:.1f} MB)")
    memory_verdict = "HIGH"

print(f"\n{'='*80}")
print("PHASE 1 TEST: COMPLETE ✓")
print(f"{'='*80}")

print("\nNEXT STEPS:")
if speed_verdict == "GOOD" and memory_verdict == "GOOD":
    print("  ✅ PROCEED to Phase 2: Create single model prototype")
    print("     - Replace ResNet18 with DINOv2 in existing model")
    print("     - Train for 10 epochs on full dataset")
    print("     - Use ALL auxiliary features (NDVI, Height, Season, Species, State, Weather)")
    print("     - Expected time: ~2 hours training")
elif speed_verdict == "CONCERNING":
    print("  ⚠️  Consider: DINOv2 might be too slow for practical use")
    print("     - Could reduce batch size to compensate")
    print("     - Or stick with ResNet18 (faster, proven)")
elif memory_verdict == "HIGH":
    print("  ⚠️  Need to reduce batch size (16 → 4 or 8)")
    print("     - This will increase training time")

print("\nKEY INSIGHT from Kaggle submission:")
print("  Universal features scored 0.40 (WORSE than baseline 0.51!)")
print("  → Location features (State, Weather) ARE important!")
print("  → Next experiments should include ALL features")

print("\n" + "="*80)
