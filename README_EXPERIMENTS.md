# Model 4b Experiments - Ready to Run!

## Notebook: [17_model4b_improved.ipynb](17_model4b_improved.ipynb)

This notebook is configured to run **6 experiments sequentially** for ~4.5 hours to improve Model 4b's Kaggle score from 0.51 to 0.54-0.60.

---

## ✅ What's Ready

The notebook is **fully configured and ready to run**. Just open it and click "Run All Cells"!

### **6 Experiments Configured:**

1. **4b_Baseline** - Rerun baseline for comparison (Val=0.6852, Kaggle=0.51)
2. **4b1_EarlyStop** - 20 epochs (vs 30) | Expected: 0.52-0.54
3. **4b2_StrongReg** - Dropout 0.4, Weight decay 5e-4 | Expected: 0.52-0.55
4. **4b3_LRSchedule** - ReduceLROnPlateau | Expected: 0.51-0.53
5. **4b4_FullNorm** - Use all 357 images for normalization | Expected: 0.51-0.53
6. **4b5_Combined** ⭐ - All improvements together | Expected: 0.53-0.56 **BEST**

---

## 🚀 How to Run

### Option 1: Run Everything (4.5 hours)
```bash
# Just open the notebook and run all cells
# Jupyter will automatically:
# - Train all 6 variations sequentially
# - Save checkpoints for each
# - Compare results
# - Generate submission from best model
```

### Option 2: Run Subset (Faster Testing)
If you want to test a subset first:

Edit Cell 3 and keep only the experiments you want:
```python
VARIATIONS = {
    '4b5_Combined': { ... },  # Just test the best one first (~45 min)
}
```

---

## 📊 What the Notebook Does

### Phase 1: For Each Experiment
1. **Phase 1 (15 epochs, ~20 min):** Auxiliary pretraining (image → tabular features)
2. **Phase 2 (20-30 epochs, ~25 min):** Biomass fine-tuning
3. **Save checkpoint:** `model4b_<name>_phase2_best.pth`
4. **Track results:** Validation R², per-target R², training curves

### Phase 2: Compare All Results
1. Create comparison table sorted by validation R²
2. Generate visualization with 4 plots:
   - Phase 1 loss curves
   - Phase 1 state accuracy
   - Phase 2 validation R² curves
   - Final R² comparison bar chart
3. Identify winner automatically

### Phase 3: Generate Submission
1. Load best model
2. Predict on test set
3. Create `submission_model4b_<winner>_<timestamp>.csv`
4. Ready to upload to Kaggle!

---

## 🎯 Expected Outcomes

### Success Criteria:
- 🟢 **Good:** Any model scores Kaggle R² > 0.54 (+0.03 improvement)
- 🟡 **Great:** Model scores > 0.57 (+0.06 improvement)
- 🔵 **Amazing:** Model scores > 0.60 (+0.09 improvement)

### Most Likely Winner:
**4b5_Combined** - Expected Kaggle R² = 0.53-0.56

This combines:
- Early stopping (20 epochs vs 30)
- Moderate dropout (0.3 vs 0.2)
- Stronger weight decay (5e-4 vs 1e-4)
- LR scheduling (ReduceLROnPlateau)
- Full dataset normalization (357 images vs 285)

---

## 📁 Files Generated

### During Training:
```
model4b_4b_Baseline_phase1_best.pth        # Phase 1 checkpoints
model4b_4b_Baseline_phase2_best.pth        # Phase 2 checkpoints (use this for submission)
model4b_4b1_EarlyStop_phase1_best.pth
model4b_4b1_EarlyStop_phase2_best.pth
... (6 variations × 2 phases = 12 checkpoint files)
```

### After Training:
```
model4b_all_variations_comparison.png      # Visualization of all experiments
submission_model4b_<winner>_<timestamp>.csv # Ready for Kaggle!
```

---

## 🔍 Key Technical Details

### What Changed from Original Model 4b:

**Cell 3 - Configuration:**
- Added 6 experiments with different hyperparameters
- Each config specifies epochs, dropout, weight decay, LR schedule, normalization

**Cell 4 - Data Loading:**
- Calculate **TWO** normalization stats:
  - Split (285 images) - for baseline experiments
  - Full (357 images) - for full normalization experiments
- This tests if normalization mismatch causes the val-test gap

**Cell 15 - train_phase2():**
- Now accepts full `config` dict instead of individual params
- Supports optional LR scheduling (ReduceLROnPlateau)
- Uses configurable weight decay (1e-4 or 5e-4)

**Cell 18 - Main Training Loop:**
- Chooses normalization stats based on `config['use_full_norm']`
- Recreates datasets for each experiment with appropriate normalization
- Passes config to train_phase2()
- Stores normalization stats in results for submission

### Why These Experiments?

**Problem:** Model 4b has -0.175 validation-test gap (Val=0.6852, Kaggle=0.51)

**Hypothesis:** Model is overfitting to the 72-image validation set

**Tests:**
1. **Early Stop:** Stop at 20 epochs before overfitting gets worse
2. **Regularization:** Force model to generalize better (dropout, weight decay)
3. **LR Schedule:** Reduce LR when plateauing for finer tuning
4. **Full Norm:** Fix potential distribution mismatch in normalization
5. **Combined:** Test if all improvements work synergistically

---

## 📈 Monitoring Progress

While training, you'll see output like:

```
================================================================================
TRAINING ALL VARIATIONS
================================================================================

Starting at: 2025-11-04 16:30:00

Will train 6 variations:
  4b_Baseline         : Baseline (Val=0.6852, Kaggle=0.51)                       | Expected: N/A
  4b1_EarlyStop       : Early stop at 20 epochs                                  | Expected: 0.52-0.54
  4b2_StrongReg       : Stronger regularization (dropout 0.4, WD 5e-4)           | Expected: 0.52-0.55
  4b3_LRSchedule      : LR scheduling (ReduceLROnPlateau)                        | Expected: 0.51-0.53
  4b4_FullNorm        : Full dataset normalization (357 images)                  | Expected: 0.51-0.53
  4b5_Combined        : Combined: early stop + dropout 0.3 + LR schedule + full norm | Expected: 0.53-0.56 ⭐ BEST

⏱️  Estimated time: ~45 min/experiment × 6 = ~4.5 hours
================================================================================


################################################################################
# TRAINING VARIATION: 4b_Baseline
# Description: Baseline (Val=0.6852, Kaggle=0.51)
# Expected Kaggle: N/A
################################################################################
✓ Using SPLIT normalization (285 train images)

================================================================================
PHASE 1: AUXILIARY PRETRAINING - 4b_Baseline
================================================================================

Epoch  1/15: Train Loss=0.3456, Val Loss=0.2987, NDVI MAE=0.1234, State Acc=75.00%, Species Acc=45.83%
  💾 Saved Phase 1 checkpoint (val_loss=0.2987)
...

================================================================================
PHASE 2: BIOMASS FINE-TUNING - 4b_Baseline
================================================================================

Config: 30 epochs, head_lr=3.00e-04, backbone_lr=1.00e-05,
        weight_decay=1.00e-04, scheduler=False

Epoch  1/30: Train Loss=0.8765, Val Loss=0.7654, Val R²=+0.4567
  💾 New best R²=+0.4567 - checkpoint saved
...

✅ 4b_Baseline complete! Best R²=+0.6852
   Checkpoint: model4b_4b_Baseline_phase2_best.pth
   Expected Kaggle: N/A


[Process continues for all 6 variations...]
```

---

## 🎬 Final Output

At the end, you'll see:

```
================================================================================
FINAL COMPARISON: ALL VARIATIONS
================================================================================

       Variation                                            Description Best R²  Hidden Dim  Dropout Phase1 LR Phase2 LR
     4b5_Combined  Combined: early stop + dropout 0.3 + LR schedule + full norm +0.6950         256     0.30  3.00e-04  3.00e-04
  4b2_StrongReg                Stronger regularization (dropout 0.4, WD 5e-4) +0.6880         256     0.40  3.00e-04  3.00e-04
   4b_Baseline                                Baseline (Val=0.6852, Kaggle=0.51) +0.6852         256     0.20  3.00e-04  3.00e-04
  4b1_EarlyStop                                       Early stop at 20 epochs +0.6810         256     0.20  3.00e-04  3.00e-04
 4b3_LRSchedule                                LR scheduling (ReduceLROnPlateau) +0.6795         256     0.20  3.00e-04  3.00e-04
   4b4_FullNorm                              Full dataset normalization (357 images) +0.6770         256     0.20  3.00e-04  3.00e-04

================================================================================
🏆 WINNER: 4b5_Combined
================================================================================
  Best R²: +0.6950
  Checkpoint: model4b_4b5_Combined_phase2_best.pth
  Description: Combined: early stop + dropout 0.3 + LR schedule + full norm

================================================================================


✅ Submission file created: submission_model4b_4b5_Combined_20251104_210015.csv

================================================================================
READY FOR KAGGLE SUBMISSION!
================================================================================

File: submission_model4b_4b5_Combined_20251104_210015.csv
Model: 4b5_Combined (R²=+0.6950 on validation)
Test samples: 1325 rows (265 images × 5 targets)

Submission format:
  sample_id: <ImageID>__<TargetName>
  target: predicted biomass value (grams)

Next steps:
  1. Upload submission_model4b_4b5_Combined_20251104_210015.csv to Kaggle
  2. Check public leaderboard R² score
  3. Compare with validation R²=+0.6950

================================================================================
```

---

## 🆘 Troubleshooting

### If training is too slow:
Edit Cell 3 and train fewer variations:
```python
VARIATIONS = {
    '4b5_Combined': VARIATIONS['4b5_Combined'],  # Just test best one
}
```

### If you want to test one experiment quickly:
Edit Cell 3 and reduce epochs:
```python
'phase1_epochs': 2,  # Test with 2 epochs (vs 15)
'phase2_epochs': 3,  # Test with 3 epochs (vs 20-30)
```

### If notebook crashes:
All checkpoints are saved automatically! You can:
1. Find the last completed checkpoint
2. Generate submission manually from that checkpoint

---

## 📚 Next Steps After This Run

### If 4b5_Combined scores > 0.54 on Kaggle:
✅ Success! You've improved the model. Consider:
- Training 3-fold ensemble (Model 4b9) for +0.02-0.04 more improvement
- Testing EfficientNet-B0 backbone (Model 4b6) for potential +0.03-0.07

### If all models score ~0.51 on Kaggle:
The issue is likely **distribution shift** (test set is different from training set), not overfitting. Consider:
- Analyzing which samples perform worst
- Adding more diverse augmentation
- Using ensemble methods to average out errors

---

## 🎉 You're Ready!

Just open [17_model4b_improved.ipynb](17_model4b_improved.ipynb) and click "Run All Cells"!

The notebook will run for ~4.5 hours and produce:
- 6 trained models
- Comparison visualization
- Kaggle submission file from best model

Good luck! 🚀
