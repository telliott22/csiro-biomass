# Universal Features K-Fold Training - Testing Guide

## What Changed and Why

### The Problem
- Previous K-Fold ensemble scored **0.50** on Kaggle (worse than baseline 0.51)
- Forum user (22nd place) revealed: **"Test dataset uses locations that aren't in the training dataset"**
- Our models learned location-specific patterns (State, Species, Weather) that don't generalize

### The Solution
Created `24_kfold_universal_features.ipynb` that:
- ✅ Removes location-specific features (State, Weather)
- ✅ Keeps universal features (NDVI, Height, Season, Daylength, **Species**)
- ✅ Uses forum user's exact weighted R² metric
- ✅ Includes DEBUG mode for quick testing

**Why Species is Universal:**
- State = location-specific (NSW conditions ≠ Tasmania conditions)
- Weather = location-specific (local rainfall patterns)
- **Species = universal!** (Ryegrass has similar biomass characteristics anywhere)

### Expected Results
- **Previous K-Fold**: 0.50 (learned location bias from State/Weather)
- **New approach**: 0.52-0.54 (should generalize to unseen locations)

---

## Quick Test (DEBUG Mode) - ~20 minutes

### Step 1: Open Notebook
```bash
jupyter notebook 24_kfold_universal_features.ipynb
```

### Step 2: Verify DEBUG Mode is ON
In **Cell 2**, check this line:
```python
DEBUG_MODE = True  # ← Should be True for testing
```

### Step 3: Run All Cells
- Click "Cell" → "Run All"
- Watch for any errors
- Expected time: **~20 minutes** (2 epochs per phase × 5 folds)

### Step 4: Check Output
After completion, verify you see:
```
✅ All 5 folds completed!
📊 Average Validation R²: X.XXXX
📂 Saved models:
   - universal_Fold1_best.pth
   - universal_Fold2_best.pth
   - universal_Fold3_best.pth
   - universal_Fold4_best.pth
   - universal_Fold5_best.pth
```

---

## Full Training (Overnight) - ~4 hours

### Only proceed if DEBUG test passed!

### Step 1: Change DEBUG Mode
In **Cell 2**, change:
```python
DEBUG_MODE = False  # ← Change True to False
```

### Step 2: Restart and Run All
- Click "Kernel" → "Restart & Run All"
- Expected time: **~4 hours**
- Training epochs: Phase 1 (15) + Phase 2 (30) per fold

### Step 3: Monitor Progress
You'll see detailed progress like:
```
Fold 1/5
Phase 1: Epoch 15/15 - Train Loss: X.XXX, Val R²: X.XXX
Phase 2: Epoch 30/30 - Train Loss: X.XXX, Val R²: X.XXX
Best Val R²: X.XXXX
```

---

## What the Notebook Does

### Model Architecture (Cell 6)
```python
class UniversalAuxiliaryModel(nn.Module):
    # ResNet18 backbone
    # Auxiliary heads: NDVI, Height, Daylength, Season, Species (5 total)
    # NO State/Weather heads!
    # Biomass prediction head
```

### Training Process (Cells 8-9)
1. **5-Fold Cross-Validation**: Data split into 5 folds
2. **Phase 1 (Auxiliary Pretraining)**:
   - Train to predict NDVI, Height, Daylength, Season, Species from images
   - Forces CNN to learn universal biomass-relevant features
   - Species classification helps model recognize plant types
   - 15 epochs (or 2 in DEBUG)
3. **Phase 2 (Fine-tuning)**:
   - Freeze backbone, train biomass head
   - 30 epochs (or 2 in DEBUG)
4. **Ensemble**: Average predictions from all 5 fold models

### Validation Metric (Cell 4)
Uses forum user's exact implementation:
```python
def weighted_r2_score(y_true, y_pred):
    # Weights: [0.1, 0.1, 0.1, 0.2, 0.5]
    # Should match Kaggle scoring exactly
```

---

## Troubleshooting

### "CUDA out of memory"
**Solution**: Reduce batch size in Cell 2:
```python
'batch_size': 16  # Try 8 or 4 if memory error
```

### "File not found: combined_tabular_features_train.csv"
**Solution**: Make sure you're running from project root directory

### Training seems stuck
- Check GPU utilization: `nvidia-smi` (if using GPU)
- Training is working if you see loss/R² values updating
- Phase 1 is faster than Phase 2

### Validation R² looks low in DEBUG mode
- This is normal! Only 2 epochs isn't enough training
- DEBUG mode is just to check for errors
- Full training (15+30 epochs) will give proper scores

---

## After Training Completes

### Next Steps:
1. **Check validation scores** - Should be ~0.68-0.70 locally
2. **Create Kaggle submission notebook** - Upload 5 model checkpoints as dataset
3. **Submit and compare**:
   - Baseline: 0.51
   - Previous K-Fold: 0.50
   - New Universal K-Fold: Expected 0.52-0.54

### If Kaggle Score is Good (≥0.52):
- This confirms location bias was the problem
- We can try further improvements:
  - Different backbones (ResNet50, EfficientNet)
  - Data augmentation tuning
  - Ensemble weighting optimization

### If Kaggle Score is Still Low (<0.52):
- May indicate other validation-test distribution shifts
- Consider: Test-time augmentation, different CV splits, or examining test set characteristics

---

## Quick Checklist

Before DEBUG test:
- [ ] Opened `24_kfold_universal_features.ipynb`
- [ ] Verified `DEBUG_MODE = True` in Cell 2
- [ ] Ready to wait ~20 minutes

Before overnight training:
- [ ] DEBUG test completed successfully
- [ ] Changed `DEBUG_MODE = False` in Cell 2
- [ ] Restarted kernel
- [ ] Ready to wait ~4 hours

After training:
- [ ] 5 model checkpoints created
- [ ] Validation R² scores look reasonable (~0.68-0.70)
- [ ] Ready to create Kaggle submission notebook

---

## Key Differences from Previous Approach

| Aspect | Previous (0.50 score) | New (Universal + Species) |
|--------|----------------------|--------------------------|
| Auxiliary heads | 7 (NDVI, Height, Weather×14, State×4, Species×15) | 5 (NDVI, Height, Daylength, Season, Species×15) |
| Location-specific | ❌ Yes (State, Weather) | ✅ No (removed State/Weather) |
| Species included | Yes (but with State/Weather bias) | Yes (without location bias) |
| Generalization | Poor (test has new locations) | Good (universal features) |
| R² metric | sklearn default | Forum user's exact implementation |
| Model complexity | Higher (7 heads + weather features) | Medium (5 heads, simpler) |

---

Good luck with the training! 🚀

**Start with DEBUG mode test first** - if it runs without errors, then proceed with overnight training.
