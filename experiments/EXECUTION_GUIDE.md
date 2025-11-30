# Experiment Execution Guide

## Quick Start

All notebooks have **DEBUG_MODE = True** by default. This runs 2 epochs per phase to verify everything works before committing to long training runs.

### Recommended Execution Order

#### Phase 1: Quick Validation (DEBUG mode, ~40 min total)

1. **Option 2** - Test DenseNet submission (~5 min)
   ```bash
   # Navigate to:
   experiments/option2_test_densenet/20_densenet_submission.ipynb
   ```
   - No training needed - just generates submission
   - Tests if validation-test gap theory holds
   - Expected: Kaggle R² < 0.51 (confirming gap)

2. **Option 4** - K-Fold CV (~20 min DEBUG)
   ```bash
   # Navigate to:
   experiments/option4_kfold_cv/21_kfold_cross_validation.ipynb
   ```
   - 🔥 **RECOMMENDED** - Most likely to improve Kaggle score
   - Trains 5 ResNet18 models (2 epochs each in DEBUG)
   - Tests ensemble approach
   - Verify it completes without errors

3. **Option 1** - EfficientNet Tuning (~15 min DEBUG)
   ```bash
   # Navigate to:
   experiments/option1_fix_efficientnet/19_efficientnet_tuned.ipynb
   ```
   - Tests 3 EfficientNet variants with higher LRs
   - Verify timm library works
   - Check if higher LRs help

#### Phase 2: Full Training (Priority Order)

Once DEBUG tests pass, set `DEBUG_MODE = False` in each notebook:

1. **Option 4 first** (~4 hours)
   - Highest expected Kaggle improvement
   - Expected: Kaggle R² = 0.53-0.55 (vs baseline 0.51)
   - Ensemble of 5 models reduces overfitting

2. **Option 2 during** (~10 min)
   - Quick submission test
   - Can run while Option 4 trains
   - Validates validation-test gap theory

3. **Option 1 optional** (~1.5 hours)
   - If Option 4 works well, might not need this
   - Academic interest / backup approach
   - Could beat baseline if EfficientNet is properly tuned

## What Each Experiment Tests

### Option 1: Fix EfficientNet Hyperparameters
- **Problem**: EfficientNet used ResNet18's LRs (too low for 1280 features)
- **Solution**: Test 3 variants with 3.3x-10x higher learning rates
- **Success**: Val R² > 0.68 (beats baseline 0.6852)

### Option 2: Test DenseNet Submission
- **Problem**: Unclear if validation scores predict Kaggle scores
- **Solution**: Submit DenseNet (Val R²=0.6605) to test
- **Success**: If Kaggle < 0.51, confirms validation-test gap

### Option 4: K-Fold Cross-Validation (RECOMMENDED)
- **Problem**: Single 80/20 split creates unrepresentative validation (72 images)
- **Solution**: 5-fold CV + ensemble
- **Success**: Kaggle R² > 0.52 (beats baseline 0.51)

## Expected Results Summary

| Experiment | Val R² (Expected) | Kaggle R² (Expected) | vs Baseline |
|------------|-------------------|----------------------|-------------|
| Baseline (ResNet18) | 0.6852 | 0.51 | - |
| Option 1 (EfficientNet Tuned) | 0.64-0.68 | 0.49-0.53 | ±0 |
| Option 2 (DenseNet Test) | 0.6605 | < 0.51 | Worse |
| **Option 4 (K-Fold)** | **0.68-0.69** | **0.53-0.55** | **+0.02-0.04** ✨ |

## Troubleshooting

### If Option 1 fails with timm errors:
```bash
pip install timm
```

### If running out of memory:
- Reduce `BATCH_SIZE` from 16 to 8
- Close other applications

### If models underperform:
- Check that `DEBUG_MODE = False` for full training
- Verify checkpoint files are being saved
- Review training logs for NaN losses

## Next Steps After Experiments

1. **Check experiments/README.md** - Update with actual results
2. **Compare all approaches** - Use validation R² and Kaggle scores
3. **Create final submission** - Use best performing approach
4. **Document findings** - Update main README.md

## Tips

- **Save checkpoints frequently** - Models are saved as `model4b_*_phase2_best.pth`
- **Monitor validation R²** - Should be positive and improving
- **Check Phase 1 State Accuracy** - Should reach 60-80% (model "sees" location)
- **Compare with baseline** - Baseline is 0.6852 val / 0.51 Kaggle

## Questions?

See [experiments/README.md](README.md) for detailed experiment descriptions.
