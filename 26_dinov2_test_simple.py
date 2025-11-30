#!/usr/bin/env python3
"""
Quick test script to verify DINOv2 works with our biomass dataset.
Uses torch.hub (requires internet) for simplicity.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import time
import numpy as np

print("="*80)
print("DINOv2 Quick Test - Phase 1 (Simplified)")
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
# Test 1: Load DINOv2 via torch.hub
# ============================================================================
print("\n2. Loading DINOv2 via torch.hub (requires internet)...")
try:
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dinov2_model = dinov2_model.to(device)
    dinov2_model.eval()
    print(f"   ✓ DINOv2 loaded successfully!")
    print(f"   Model size: {sum(p.numel() for p in dinov2_model.parameters()):,} parameters")
except Exception as e:
    print(f"   ✗ Failed to load DINOv2: {e}")
    print("\n   This is expected if internet is disabled or torch.hub has issues.")
    print("   We can still proceed with local checkpoint in actual training.")
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
transform = transforms.Compose([
    transforms.Resize((224, 224)),
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
    img_tensor = transform(img).unsqueeze(0).to(device)

    # ResNet18 features
    start = time.time()
    with torch.no_grad():
        resnet_feat = resnet18_backbone(img_tensor).flatten()
    resnet_time = time.time() - start
    resnet_times.append(resnet_time)
    resnet_features_list.append(resnet_feat.cpu().numpy())

    # DINOv2 features
    start = time.time()
    with torch.no_grad():
        dinov2_feat = dinov2_model(img_tensor).flatten()
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
print(f"     - Feature std: {np.std(resnet_features_list[0]):.3f}")

print(f"   DINOv2:")
print(f"     - Feature dim: {dinov2_features_list[0].shape[0]}")
print(f"     - Avg inference time: {np.mean(dinov2_times)*1000:.1f}ms")
print(f"     - Feature range: [{np.min(dinov2_features_list[0]):.3f}, {np.max(dinov2_features_list[0]):.3f}]")
print(f"     - Feature std: {np.std(dinov2_features_list[0]):.3f}")

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

# Estimate batch size impact
print(f"\n   Estimated max batch size:")
print(f"     - ResNet18: 16 (current)")
print(f"     - DINOv2: ~{int(16 * resnet_mem / dinov2_mem)} (estimated)")

# ============================================================================
# Summary & Recommendation
# ============================================================================
print("\n" + "="*80)
print("SUMMARY & RECOMMENDATION")
print("="*80)

print(f"\n✓ DINOv2 loaded successfully")
print(f"✓ Feature extraction works ({dinov2_features_list[0].shape[0]}-dim features)")
print(f"✓ Compatible with our 224×224 images")

if speedup < 2.0:
    print(f"✓ Inference speed acceptable ({speedup:.1f}x slower than ResNet18)")
    speed_verdict = "GOOD"
elif speedup < 3.0:
    print(f"⚠️  Inference moderately slower ({speedup:.1f}x slower than ResNet18)")
    speed_verdict = "OK"
else:
    print(f"❌ Inference too slow ({speedup:.1f}x slower than ResNet18)")
    speed_verdict = "TOO_SLOW"

if dinov2_mem < 500:
    print(f"✓ Memory usage acceptable ({dinov2_mem:.1f} MB)")
    memory_verdict = "GOOD"
else:
    print(f"⚠️  Memory usage high ({dinov2_mem:.1f} MB) - will need smaller batch size")
    memory_verdict = "HIGH"

print(f"\n{'='*80}")
print("PHASE 1 TEST: COMPLETE ✓")
print(f"{'='*80}")

print("\nKEY FINDINGS:")
print(f"  1. DINOv2 feature dim: 768 (vs ResNet18: 512)")
print(f"  2. Inference speed: {speedup:.1f}x slower")
print(f"  3. Model size: {dinov2_mem/resnet_mem:.1f}x larger")

print("\nNEXT STEPS:")
if speed_verdict in ["GOOD", "OK"] and memory_verdict in ["GOOD", "HIGH"]:
    print("  ✅ PROCEED to Phase 2: Create single model prototype")
    print()
    print("  Configuration for prototype:")
    print("    - Backbone: DINOv2 (dinov2_vitb14)")
    print("    - Feature dim: 768")
    print("    - Batch size: 8 (reduced from 16)")
    print("    - Auxiliary features: ALL (NDVI, Height, Season, Species, State, Weather)")
    print("    - Training: 10 epochs on full dataset")
    print("    - Expected time: ~2-3 hours")
    print()
    print("  Why ALL features:")
    print("    - Universal features only scored 0.40 on Kaggle")
    print("    - Baseline with all features scored 0.51")
    print("    - Location features (State, Weather) are important!")
else:
    print("  ⚠️  DINOv2 may not be suitable:")
    if speed_verdict == "TOO_SLOW":
        print("     - Too slow for practical use")
    if memory_verdict == "HIGH":
        print("     - Memory usage too high")
    print("     - Recommend sticking with ResNet18")

print("\n" + "="*80)
