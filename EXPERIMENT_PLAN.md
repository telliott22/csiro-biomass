# Model 4b Improvement Experiments

## Summary

**Current Status:**
- Model 4b (ResNet18, 30 epochs): Val R²=+0.6852, **Kaggle R²=+0.51**
- Model 1 (ResNet18, 10 epochs): Val R²=+0.6352, Kaggle R²=+0.48
- **Problem:** Validation-test gap of -0.175 (too large!)

**Goal:** Improve Kaggle score from 0.51 to 0.54-0.58

---

## Key Insights

### 1. Why the Large Val-Test Gap?

**The validation set (72 images) is NOT representative of test set.**

Evidence:
- More epochs → Better val R² (+0.6352 → +0.6852) ✅
- More epochs → Worse gap (-0.155 → -0.175) ❌
- Best epoch at 29/30 (keeps "improving" on validation)

**Conclusion:** Model is learning validation-specific patterns that don't generalize!

### 2. Why ResNet50 Failed

From notebook 11:
- ResNet18 (11M params): Val R²=+0.6352 ✅
- ResNet50 (25M params): Val R²=+0.5529 ❌

**Conclusion:** More parameters = worse with only 285 images

**Therefore:**
- ❌ Don't try ResNet34 (21M params) - will likely score ~0.59-0.61
- ✅ Try EFFICIENT architectures (EfficientNet, MobileNet)

### 3. Research on Small Datasets (2025)

**Best for small datasets:**
1. **EfficientNet-B0** ⭐ - Designed for efficiency, excellent transfer learning
2. **MobileNetV3** - Lightweight, works well with limited data
3. **DenseNet121** - Dense connections, efficient feature reuse

**Bad for small datasets:**
- Vision Transformers (ViT, Swin) - Need 1000+ images
- Large CNNs (ResNet50+) - Too many parameters

---

## Recommended Experiments

### Priority 1: Combined Improvements (Model 4b5) 🟢 LOW RISK

**Changes:**
- Phase 2 epochs: 20 (vs 30)
- Dropout: 0.3 (vs 0.2)
- Weight decay: 5e-4 (vs 1e-4)
- LR scheduling: ReduceLROnPlateau (factor=0.5, patience=3)
- Normalization: Full dataset (357 images vs 285)

**Hypothesis:** All improvements work synergistically

**Expected Kaggle R²:** 0.53-0.55 (+0.02 to +0.04)

**Time:** ~45 minutes

**How to run:**
```bash
# Edit notebook 13_model4b_final_training.ipynb
# Change VARIATIONS to only train "A_Baseline" with these settings:
VARIATIONS = {
    'A_Improved': {
        'phase1_lr': 3e-4,
        'phase2_lr': 3e-4,
        'phase2_backbone_lr': 1e-5,
        'hidden_dim': 256,
        'dropout': 0.3,  # CHANGED from 0.2
        'description': 'Combined improvements'
    }
}

# Change epochs:
PHASE2_EPOCHS = 20  # CHANGED from 30

# Change weight decay:
# In train_phase2(), change weight_decay=1e-4 to weight_decay=5e-4

# Add scheduler to train_phase2():
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)
# After calculating val_r2:
scheduler.step(val_r2)

# Calculate normalization stats from FULL dataset (before train/val split)
```

---

### Priority 2: EfficientNet-B0 Backbone (Model 4b6) 🔴 HIGH RISK / HIGH REWARD

**Changes:**
- Backbone: ResNet18 → EfficientNet-B0 (11M → 5M params)
- Everything else: same as baseline

**Hypothesis:** More efficient architecture → better with small dataset

**Expected Kaggle R²:** 0.54-0.58 (+0.03 to +0.07) ⭐ **BEST POTENTIAL**

**Time:** ~45 minutes

**Requirements:**
```bash
pip install timm
```

**How to run:**
```python
# Replace ResNet18 backbone with EfficientNet-B0
import timm

# In AuxiliaryPretrainedModel.__init__():
# Replace:
# self.resnet = models.resnet18(pretrained=True)
# self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

# With:
model = timm.create_model('efficientnet_b0', pretrained=True)
self.backbone = nn.Sequential(
    model.conv_stem,
    model.bn1,
    model.act1,
    model.blocks,
    model.conv_head,
    model.bn2,
    model.act2,
    nn.AdaptiveAvgPool2d((1, 1))
)

# Update feature dimension:
# In auxiliary heads and biomass head, replace 512 → 1280
```

---

### Priority 3: Early Stopping Only (Model 4b1) 🟢 LOW RISK

**Changes:**
- Phase 2 epochs: 20 (vs 30)
- Everything else: same as baseline

**Hypothesis:** Fewer epochs → less overfitting to validation

**Expected Kaggle R²:** 0.52-0.54 (+0.01 to +0.03)

**Time:** ~30 minutes

**How to run:**
```bash
# Edit notebook 13, change PHASE2_EPOCHS = 20
```

---

### Priority 4: Full Dataset Normalization (Model 4b4) 🟢 LOW RISK

**Changes:**
- Normalization: Use all 357 images (vs 285 train split)
- Everything else: same as baseline

**Hypothesis:** Correct normalization → better test set match

**Expected Kaggle R²:** 0.51-0.53 (+0.00 to +0.02)

**Time:** ~45 minutes

**Current stats:**
```python
# Split (285 images):
Dry_Green_g: mean=27.49g, std=26.19g

# Full (357 images):
Dry_Green_g: mean=26.62g, std=25.40g

# Difference: -0.87g mean, -0.79g std
```

**How to run:**
```python
# Calculate stats BEFORE train/val split:
target_means = torch.tensor([train_enriched[col].mean() for col in TARGET_COLS])
target_stds = torch.tensor([train_enriched[col].std() for col in TARGET_COLS])

# Then do train/val split
train_data, val_data = train_test_split(train_enriched, ...)
```

---

### Priority 5: 3-Fold Ensemble (Model 4b9) 🔵 MOST ROBUST

**Changes:**
- Train 3 models with random seeds: 42, 123, 456
- Different train/val splits for each
- Average predictions from all 3

**Hypothesis:** Ensemble is robust to validation set bias

**Expected Kaggle R²:** 0.56-0.61 (+0.05 to +0.10) ⭐ **HIGHEST SCORE**

**Time:** ~2.5 hours (3 × 50 minutes)

**How to run:**
```python
# Train Model 4b5 (combined) 3 times with different seeds:
for seed in [42, 123, 456]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_data, val_data = train_test_split(train_enriched, random_state=seed)
    # Train model...
    # Save as model4b9_fold{i}_best.pth

# At inference:
predictions = []
for i in range(3):
    model.load_state_dict(torch.load(f'model4b9_fold{i}_best.pth'))
    pred = model(images)
    predictions.append(pred)

# Average
final_pred = torch.stack(predictions).mean(dim=0)
```

---

## Testing Order

### Phase 1: Quick Tests [~2 hours]
1. **Model 4b5** (Combined) - 45 min
2. **Model 4b1** (Early Stop) - 30 min
3. **Model 4b4** (Full Norm) - 45 min

**Decision point:** If any model scores > 0.53 on Kaggle, continue to Phase 2

### Phase 2: Architecture [~1 hour]
4. **Model 4b6** (EfficientNet) - 45 min
   - **Install timm first:** `pip install timm`
   - Highest potential but requires code changes

### Phase 3: Ensemble [~2.5 hours]
5. **Model 4b9** (Ensemble) - 2.5 hours
   - Only if Phase 1-2 shows improvement
   - Most robust but most expensive

**Total time:** 2-5.5 hours depending on results

---

## Success Criteria

- 🟢 **Good:** Kaggle R² > 0.54 (+0.03)
- 🟡 **Great:** Kaggle R² > 0.57 (+0.06)
- 🔵 **Amazing:** Kaggle R² > 0.60 (+0.09)

---

## Quick Start (Easiest Test)

**Test Model 4b1 (Early Stop) in 30 minutes:**

1. Open `13_model4b_final_training.ipynb`
2. Change line: `PHASE2_EPOCHS = 30` → `PHASE2_EPOCHS = 20`
3. Change line: `VARIATIONS = { ... }` → Keep only `'A_Baseline'`
4. Run all cells
5. Generate submission from `model4b_A_Baseline_phase2_best.pth`
6. Submit to Kaggle
7. **Expected score:** 0.52-0.54

If this works, try Model 4b5 (Combined) next!

---

## Files to Modify

**For Model 4b1, 4b4, 4b5:**
- Edit: `13_model4b_final_training.ipynb`
- No new files needed

**For Model 4b6 (EfficientNet):**
- Create: `17_model4b6_efficientnet.ipynb` (copy from notebook 13)
- Install: `pip install timm`
- Modify: Model architecture code

**For Model 4b9 (Ensemble):**
- Create: `18_model4b9_ensemble.ipynb`
- Train 3 models sequentially
- Create: `18b_ensemble_submission.ipynb` for averaging

---

## Next Actions

**Recommendation:** Start with Model 4b1 (early stopping) as a quick test:
- Lowest risk
- Only 1 line change
- 30 minutes to train
- Will tell us if fewer epochs helps

**If Model 4b1 improves:** Try Model 4b5 (combined) next
**If Model 4b1 doesn't improve:** Try Model 4b6 (EfficientNet)
**If nothing improves:** The issue is distribution shift (not overfitting)
