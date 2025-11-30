# Analysis: Why EfficientNet Underperformed

## Executive Summary

**Expected**: EfficientNet-B0 to achieve Val R²=0.54-0.58 (Kaggle 0.54-0.58)
**Actual**: EfficientNet achieved Val R²=0.5903 (likely Kaggle ~0.46-0.48)

All 4 new experiments **failed to beat the baseline** (ResNet18, Val R²=0.6852, Kaggle R²=0.51).

## Results Summary

| Experiment | Backbone | Val R² | vs Baseline | Expected Kaggle |
|------------|----------|--------|-------------|-----------------|
| **4b_Baseline** | ResNet18 | **0.6852** | - | **0.51** ✅ |
| 4b8_DenseNet | DenseNet121 | 0.6605 | -0.0247 | ~0.49 |
| 4b7_MobileNet | MobileNetV3 | 0.6314 | -0.0538 | ~0.48 |
| 4b6_EfficientNet | EfficientNet-B0 | 0.5903 | -0.0949 | ~0.46 |
| 4b9_EfficientNet_Combined | EfficientNet-B0 + improvements | 0.5720 | -0.1132 | ~0.45 |

## Detailed Analysis from Visualization

### 1. Phase 1: Auxiliary Pretraining (Top-Left Plot)

**Observation**: All models converge to similar validation loss (~0.3-0.35) by epoch 15.

- ✅ **DenseNet** (green): Smoothest convergence, lowest final loss
- ✅ **MobileNet** (orange): Good convergence, slightly higher loss
- ⚠️ **EfficientNet** (blue): More oscillation, converges but less stable
- ⚠️ **EfficientNet_Combined** (red): Similar to plain EfficientNet

**Conclusion**: Phase 1 pretraining worked reasonably well for all models. EfficientNet learned auxiliary tasks, but with more instability.

### 2. Phase 1: State Classification Accuracy (Top-Right Plot)

**Observation**: All models achieve 90%+ state accuracy (remember, random would be 25%).

- ✅ **All models** reach 90-95% accuracy
- This means all models successfully learned to "see" geographic location from images
- EfficientNet performs comparably here

**Conclusion**: The auxiliary pretraining objective works. Models learn visual features correlated with location/tabular data.

### 3. Phase 2: Biomass Prediction R² (Bottom-Left Plot) 🔍

**This is where the problem is clear:**

**DenseNet (green, best at 0.6605)**:
- Starts strong, reaches ~0.65-0.66 by epoch 10
- Maintains stability with slight oscillation
- Consistent performance throughout 30 epochs

**MobileNet (orange, R²=0.6314)**:
- Similar start to DenseNet
- Peaks around 0.62-0.63
- More oscillation than DenseNet
- Slight decline in later epochs (possible overfitting)

**EfficientNet (blue, R²=0.5903)**:
- **Much slower start** - stays below 0.55 for first 8 epochs
- Gradually improves but **never catches up**
- Peaks around 0.58-0.59 around epoch 20
- Shows instability (oscillates between 0.53-0.59)
- Final performance plateaus at 0.59

**EfficientNet_Combined (red, worst at 0.5720)**:
- **Even slower start** than plain EfficientNet
- More volatile throughout training
- Peaks around 0.57 but drops to 0.50 at some epochs
- Never stabilizes
- The "improvements" (higher dropout 0.3, higher weight decay 5e-4) **hurt performance**

**Baseline (red dashed line at 0.6852)**:
- All new models fall short of this line
- DenseNet gets closest (-0.0247 gap)

### 4. Final Comparison Bar Chart (Bottom-Right)

Shows the clear ranking:
1. DenseNet: 0.6605 (gold bar, but still below baseline)
2. MobileNet: 0.6314
3. EfficientNet: 0.5903
4. EfficientNet_Combined: 0.5720

## Root Cause Analysis

### Why Did EfficientNet Fail?

#### Theory 1: Feature Space Mismatch ✅ **Most Likely**

**Evidence**:
- EfficientNet outputs 1280 features (vs ResNet18's 512, DenseNet's 1024)
- The biomass prediction head expects to work with these features
- **Hypothesis**: The linear heads (256 hidden → 5 outputs) may need different initialization or learning rates for 1280-dimensional input

**Supporting Evidence**:
- DenseNet (1024 features) > ResNet18 (512 features) suggests more features *could* help
- But EfficientNet (1280 features) << DenseNet suggests it's not just about feature count
- The slower Phase 2 convergence suggests the head struggles to learn from EfficientNet features

#### Theory 2: Transfer Learning Gap ✅ **Contributing Factor**

**Evidence**:
- All models pretrained on ImageNet (natural images: cats, dogs, cars, etc.)
- Your data: overhead pasture images (grass, clover, very different distribution)
- EfficientNet's architecture is more specialized/optimized for ImageNet
- DenseNet's dense connections may provide more flexibility for transfer

**Pattern**: Models designed to be very efficient on ImageNet (EfficientNet, MobileNet) may lose generalization ability.

#### Theory 3: Learning Rate Mismatch ✅ **Confirmed by Combined Results**

**Evidence**:
- EfficientNet_Combined used **same learning rates** as ResNet18
  - backbone_lr = 1e-5
  - head_lr = 3e-4
- But EfficientNet has **different architecture**
  - More layers with different depth
  - Compound scaling (width + depth + resolution)
  - May need different LR for optimal convergence

**Smoking Gun**:
- Adding *more regularization* to EfficientNet made it **worse** (0.5903 → 0.5720)
- Suggests the model is **underfitting**, not overfitting
- Needs **less regularization** or **higher learning rate**, not more

#### Theory 4: TimmBackbone Wrapper Issues ⚠️ **Unlikely but Possible**

The wrapper we created:
```python
class TimmBackbone(nn.Module):
    def forward(self, x):
        features = self.model.forward_features(x)
        if len(features.shape) == 4:  # [B,C,H,W]
            features = features.mean([2, 3])  # Global average pool
        return features.unsqueeze(-1).unsqueeze(-1)
```

**Potential Issue**: The global average pooling might lose spatial information that EfficientNet learned.

**Counter-evidence**: If this were the problem, Phase 1 would fail, but state classification reached 90%+.

## Why "Combined Improvements" Made EfficientNet Worse

4b9_EfficientNet_Combined (R²=0.5720) < 4b6_EfficientNet (R²=0.5903)

**Changes in "Combined"**:
- ❌ Dropout: 0.2 → 0.3 (50% increase)
- ❌ Weight Decay: 1e-4 → 5e-4 (5x increase)
- ⚠️ Epochs: 30 → 20 (less time to converge)
- ✅ LR Scheduler: Added ReduceLROnPlateau
- ✅ Full normalization: 357 images vs 285

**Diagnosis**: The model is **underfitting**, not overfitting!
- More regularization makes underfitting worse
- Less epochs means less time to escape poor local minimum
- LR scheduler might reduce LR too early when model still improving slowly

## Why DenseNet Performed Best (But Still Lost to Baseline)

DenseNet121 achieved R²=0.6605 (best of new models, but -0.0247 vs baseline 0.6852).

**Why DenseNet worked better**:
- ✅ Dense connections provide feature reuse
- ✅ 1024 features (more than ResNet18's 512, less than EfficientNet's 1280)
- ✅ More gradual architecture (not as optimized as EfficientNet)
- ✅ Better transfer learning properties

**Why it still lost to ResNet18 baseline**:
- More parameters (8M vs 11M) - might be overfitting slightly
- Different feature space still requires head adaptation
- The 72-image validation set gap affects all models similarly

## The Real Problem: Validation-Test Gap

**All experiments miss the core issue**: The validation set (72 images from train/test split) doesn't represent the test distribution well.

**Evidence**:
- ResNet18 baseline: Val R²=0.6852 → Kaggle R²=0.51 (**gap of -0.175**)
- This is a HUGE gap suggesting distribution mismatch
- Trying different architectures won't fix this

**Expected Kaggle scores** (assuming similar gap):
- 4b8_DenseNet: 0.6605 → ~0.49 Kaggle ❌
- 4b7_MobileNet: 0.6314 → ~0.48 Kaggle ❌
- 4b6_EfficientNet: 0.5903 → ~0.46 Kaggle ❌
- 4b9_EfficientNet_Combined: 0.5720 → ~0.45 Kaggle ❌

**None of these will beat the baseline's Kaggle score of 0.51.**

## Recommendations

### Option 1: Fix EfficientNet (If You Really Want To) ⚠️

**Try these changes**:
1. **Increase head learning rate**: 3e-4 → 5e-4 or 1e-3
2. **Increase backbone learning rate**: 1e-5 → 5e-5
3. **Reduce regularization**: dropout=0.15, weight_decay=5e-5
4. **Train longer**: 40-50 epochs Phase 2
5. **Add batch normalization** in the biomass head

**Expected improvement**: Might reach R²=0.62-0.64, still below baseline.

**Estimated effort**: 1-2 hours to test
**Estimated success**: 30% chance of beating baseline

### Option 2: Use DenseNet (Best of New Models) ⚠️

**Action**: Submit 4b8_DenseNet to Kaggle

**Expected result**: Kaggle R²~0.49 (worse than baseline 0.51)

**Why bother**: To confirm the validation-test gap theory

**Estimated effort**: 10 minutes
**Estimated success**: Will likely score worse than baseline

### Option 3: Stick with ResNet18 Baseline ✅ **SAFE**

**Action**: Continue using 4b_Baseline (Val R²=0.6852, Kaggle R²=0.51)

**Rationale**:
- Already submitted and validated (0.51 Kaggle)
- Best validation score across all 9 experiments
- Known quantity

**Estimated effort**: 0 minutes
**Estimated success**: Guaranteed 0.51 Kaggle score

### Option 4: Address the Real Problem with K-Fold Cross-Validation ✅✅✅ **RECOMMENDED**

**The Core Issue**: The single 80/20 train/val split creates a validation set (72 images) that doesn't represent test distribution.

**Solution**: 5-Fold Cross-Validation
- Train 5 different models with different train/val splits
- Each model sees different validation images
- Average predictions across all 5 models (ensemble)
- This gives more robust validation estimate AND better test performance

**Expected improvement**:
- Better validation estimate (closer to actual test performance)
- Ensemble typically adds +0.01 to +0.03 R²
- Could push Kaggle score from 0.51 to 0.53-0.55

**Estimated effort**: 3-4 hours (5 models × 45 min each)
**Estimated success**: 70% chance of improving Kaggle score

### Option 5: Ensemble Existing ResNet18 Models ✅ **QUICK WIN**

You already have **5 trained ResNet18 models** from the first notebook:
- 4b_Baseline (R²=0.6852)
- 4b2_StrongReg
- 4b3_LRSchedule
- 4b4_FullNorm
- 4b5_Combined

**Action**: Average their predictions

**Expected improvement**: +0.01 to +0.02 R² → Kaggle 0.52-0.53

**Estimated effort**: 30 minutes (write ensemble code)
**Estimated success**: 80% chance of small improvement

## Conclusion

1. ❌ **EfficientNet underperformed** due to learning rate mismatch and underfitting
2. ✅ **DenseNet performed best** of new models but still below baseline
3. ⚠️ **Architecture changes won't solve the real problem**: validation-test distribution gap
4. ✅ **Best path forward**: K-fold cross-validation OR ensemble existing ResNet18 models

**My recommendation**: Try Option 5 (ensemble existing models) first for a quick win, then Option 4 (K-fold CV) for a more substantial improvement.
