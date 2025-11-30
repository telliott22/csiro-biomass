# Model 4b Improvement Experiments

## Problem Statement

Original Model 4b (ResNet18):
- **Validation R²**: +0.6852
- **Kaggle R²**: +0.51
- **Gap**: -0.175 (validation-test mismatch)

New backbones (EfficientNet, MobileNet, DenseNet) **underperformed** due to using ResNet18's hyperparameters.

## Experiment Structure

### Option 1: Fix EfficientNet with Proper Hyperparameters
**Notebook**: `option1_fix_efficientnet/19_efficientnet_tuned.ipynb`

**Hypothesis**: EfficientNet (1280 features) needs higher learning rates than ResNet18 (512 features)

**Experiments**:
| Variant | Head LR | Backbone LR | Dropout | Weight Decay | Epochs P2 |
|---------|---------|-------------|---------|--------------|-----------|
| HigherLR | 1e-3 | 5e-5 | 0.15 | 5e-5 | 40 |
| VeryHighLR | 2e-3 | 1e-4 | 0.15 | 5e-5 | 40 |
| WithBatchNorm | 1e-3 | 5e-5 | 0.15 | 5e-5 | 40 |

**Expected**: R²=0.64-0.68 (might beat baseline!)
**Time**: ~1.5 hours full training, 15 min DEBUG

### Option 2: Test DenseNet Submission
**Notebook**: `option2_test_densenet/20_densenet_submission.ipynb`

**Hypothesis**: Validation-test gap affects all models similarly

**Action**: Submit DenseNet (Val R²=0.6605) to Kaggle
**Expected Kaggle**: ~0.49 (worse than baseline 0.51)
**Purpose**: Confirms the gap theory
**Time**: 10 minutes

### Option 4: K-Fold Cross-Validation (RECOMMENDED)
**Notebook**: `option4_kfold_cv/21_kfold_cross_validation.ipynb`

**Hypothesis**: Single 80/20 split creates unrepresentative validation set

**Solution**: 5-fold stratified CV
- Train 5 ResNet18 models (proven best architecture)
- Each sees different validation data
- Ensemble predictions for robustness

**Expected**: Kaggle R²=0.53-0.55 (improves from 0.51)
**Time**: ~4 hours full training, 20 min DEBUG

## Results Tracking

### Baseline (From Previous Experiments)

| Model | Backbone | Val R² | Kaggle R² | Gap |
|-------|----------|--------|-----------|-----|
| 4b_Baseline | ResNet18 | 0.6852 | 0.51 | -0.175 |
| 4b8_DenseNet | DenseNet121 | 0.6605 | TBD | TBD |
| 4b6_EfficientNet | EfficientNet-B0 | 0.5903 | TBD | TBD |

### Option 1 Results (EfficientNet Tuned)

| Variant | Val R² | Kaggle R² | vs Baseline | Status |
|---------|--------|-----------|-------------|--------|
| HigherLR | TBD | TBD | TBD | Not run |
| VeryHighLR | TBD | TBD | TBD | Not run |
| WithBatchNorm | TBD | TBD | TBD | Not run |

### Option 2 Results (DenseNet Test)

| Model | Val R² | Kaggle R² | Gap | Theory Confirmed? |
|-------|--------|-----------|-----|-------------------|
| DenseNet | 0.6605 | TBD | TBD | TBD |

### Option 4 Results (K-Fold CV)

| Fold | Val R² | Train/Val Split |
|------|--------|-----------------|
| Fold 1 | TBD | TBD |
| Fold 2 | TBD | TBD |
| Fold 3 | TBD | TBD |
| Fold 4 | TBD | TBD |
| Fold 5 | TBD | TBD |
| **Ensemble** | **TBD** | **TBD (Kaggle)** |

## Execution Plan

### Phase 1: Quick Validation (DEBUG mode, ~40 min)
1. Test Option 2 (5 min) - Verify DenseNet submission works
2. Test Option 4 (20 min) - Verify K-fold works
3. Test Option 1 (15 min) - Verify EfficientNet tuning works

### Phase 2: Full Training (Priority order)
1. **Option 4 first** (~4 hours) - Most likely to improve Kaggle score
2. **Option 2 during** (~10 min) - Quick Kaggle submission test
3. **Option 1 optional** (~1.5 hours) - If time permits / academic interest

## Success Criteria

- ✅ **Option 1 Success**: EfficientNet Val R² > 0.68
- ✅ **Option 2 Success**: Confirms gap (Kaggle < 0.51)
- ✅ **Option 4 Success**: Kaggle R² > 0.52 (beats baseline)

## Final Recommendation

After all experiments, recommend the approach with best **Kaggle score** (not validation score).
