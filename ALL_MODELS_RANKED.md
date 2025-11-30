# Complete Model Rankings - All Models Trained

## Summary Table

| Rank | Model | Validation R² | Kaggle R² | Can Submit? | Checkpoint | Epochs | Notes |
|------|-------|---------------|-----------|-------------|------------|--------|-------|
| 🥇 **1** | **Model 4b A_Baseline** | **+0.6852** | **+0.51** | ✅ Yes | model4b_A_Baseline_phase2_best.pth | 30 | ⚠️ Large gap! |
| 🥈 **2** | **Model 4b C_LargerHead** | **+0.6711** | Not tested | ✅ Yes | model4b_C_LargerHead_phase2_best.pth | 30 | 512 hidden |
| 🥉 **3** | **Model 4 Teacher** | **+0.6642** | N/A | ❌ No | Model_4_Teacher_best.pth | 10 | Needs tabular |
| **4** | **Model 4b B_HigherLR** | **+0.6641** | Not tested | ✅ Yes | model4b_B_HigherLR_phase2_best.pth | 30 | Higher LR |
| **5** | **Model 1 Simple** | **+0.6352** | **Testing next!** | ✅ Yes | Model_1_Simple_best.pth | 10 | ⭐ Recommended |
| **6** | **Model 3 Multimodal** | **+0.6055** | N/A | ❌ No | Model_3_Multimodal_best.pth | 10 | Needs tabular |
| **7** | **Model 2 ColorJitter** | **+0.5584** | Not tested | ✅ Yes | Model_2_ColorJitter_best.pth | 10 | Worse than no CJ |
| **8** | **Model 4 Student** | **+0.5553** | Not tested | ✅ Yes | Model_4_Student_best.pth | 10 | Distillation |
| **9** | **Model 5 ResNet50** | **+0.5529** | Not tested | ✅ Yes | Model_5_ResNet50_best.pth | 10 | Larger backbone |

---

## Detailed Model Information

### 1. Model 4b A_Baseline - Auxiliary Pretrained (Winner on Validation)

**Architecture:**
- ResNet18 backbone
- Two-phase training:
  - Phase 1: Train to predict NDVI, height, weather, state, species from images (15 epochs)
  - Phase 2: Fine-tune for biomass (30 epochs)
- 5 auxiliary heads (512→1 each for NDVI/height, 512→14 for weather, etc.)
- Biomass head: 512→256→ReLU→Dropout(0.2)→256→5

**Training:**
- Phase 1 epochs: 15
- Phase 2 epochs: 30
- Total training time: ~3 hours
- Learning rates: 3e-4 (phase 1), 3e-4 head + 1e-5 backbone (phase 2)

**Performance:**
- Validation R²: **+0.6852** (best!)
- Kaggle R²: **+0.51** (disappointing!)
- Gap: **-0.175** (overfitting suspected)

**Per-target validation R² (at best epoch):**
- Dry_Green_g: +0.6231
- Dry_Dead_g: +0.5489
- Dry_Clover_g: +0.3892
- GDM_g: +0.7145
- Dry_Total_g: +0.8234

**Key insight:**
- Model learned to "see" location (88% state accuracy in phase 1!)
- But may have overfit during 30 epochs of phase 2

**Can submit?** ✅ Yes (image-only at inference)

---

### 2. Model 4b C_LargerHead - Auxiliary Pretrained (Larger Hidden Layer)

**Architecture:**
- Same as Model 4b A_Baseline
- **Difference:** Hidden dim = 512 (vs 256)
- Biomass head: 512→**512**→ReLU→Dropout(0.25)→512→5

**Training:**
- Phase 1: 15 epochs
- Phase 2: 30 epochs
- Dropout: 0.25 (slightly more than A_Baseline's 0.2)

**Performance:**
- Validation R²: **+0.6711**
- Kaggle R²: Not tested yet
- Slightly worse than A_Baseline, possibly due to higher dropout

**Can submit?** ✅ Yes

---

### 3. Model 4 Teacher - Multimodal (Best Multimodal)

**Architecture:**
- ResNet18 + weather encoder + tabular encoder
- Image: 512 features
- Weather: 14→64 features
- Tabular: NDVI/height + state_emb + species_emb → 32 features
- Fusion: 608→256→5

**Training:**
- Epochs: 10
- Learning rate: 3e-4

**Performance:**
- Validation R²: **+0.6642**
- Kaggle R²: N/A (cannot submit)

**Why?** Cannot submit because needs tabular features at test time!

**Can submit?** ❌ No (requires weather, NDVI, state, species at inference)

---

### 4. Model 4b B_HigherLR - Auxiliary Pretrained (Higher Learning Rate)

**Architecture:**
- Same as Model 4b A_Baseline
- **Difference:** Higher learning rates

**Training:**
- Phase 1 LR: 5e-4 (vs 3e-4)
- Phase 2 LR: 5e-4 head, 5e-5 backbone (vs 3e-4 and 1e-5)
- Dropout: 0.3 (vs 0.2)

**Performance:**
- Validation R²: **+0.6641**
- Kaggle R²: Not tested yet
- Slightly worse than A_Baseline

**Can submit?** ✅ Yes

---

### 5. Model 1 Simple - ResNet18 Baseline ⭐ RECOMMENDED NEXT!

**Architecture:**
- ResNet18 backbone (pretrained on ImageNet)
- Simple FC head: 512→256→ReLU→Dropout(0.2)→256→5
- **No auxiliary heads**
- **No ColorJitter** (learned this hurts!)

**Training:**
- Epochs: **10 only** (vs 30 for Model 4b)
- Learning rate: 3e-4
- Weight decay: 1e-4
- Training time: ~40 minutes

**Performance:**
- Validation R²: **+0.6352**
- Kaggle R²: **Ready to test!**
- Validation progressed steadily: 0.25→0.47→0.57→0.64

**Why it might do better than Model 4b on Kaggle:**
- ✅ Simpler (less parameters to overfit)
- ✅ Trained for fewer epochs (10 vs 30)
- ✅ No complex auxiliary heads
- ✅ Validation curve looks healthy (no bouncing)

**Predictions on test image:**
- Lower biomass than Model 4b (-22% on average)
- Might be more conservative/accurate

**Can submit?** ✅ Yes - **DO THIS NEXT!**

---

### 6. Model 3 Multimodal - Full Multimodal

**Architecture:**
- Same as Model 4 Teacher
- ResNet18 + weather + tabular fusion

**Training:**
- Epochs: 10

**Performance:**
- Validation R²: **+0.6055**

**Can submit?** ❌ No (needs tabular features)

---

### 7. Model 2 ColorJitter - ResNet18 + ColorJitter

**Architecture:**
- ResNet18 baseline
- **With ColorJitter augmentation** (brightness/contrast/saturation ±0.3)

**Training:**
- Epochs: 10
- Same as Model 1 but with ColorJitter

**Performance:**
- Validation R²: **+0.5584**
- **0.0768 worse than Model 1** (no ColorJitter)

**Key learning:** ColorJitter **hurts** this task! Green/brown colors are important features.

**Can submit?** ✅ Yes (but not recommended)

---

### 8. Model 4 Student - Knowledge Distillation

**Architecture:**
- Same as Model 1 Simple (ResNet18 baseline)
- **Trained via distillation from Model 4 Teacher**

**Training:**
- Epochs: 10
- Loss: α×hard_loss + β×soft_loss + γ×feature_loss
- Temperature: 4.0

**Performance:**
- Validation R²: **+0.5553**
- **Worse than simple baseline** (Model 1: +0.6352)

**Why?** Distillation didn't help - dataset may be too small, or 10 epochs insufficient.

**Can submit?** ✅ Yes

---

### 9. Model 5 ResNet50 - Larger Backbone

**Architecture:**
- ResNet50 (vs ResNet18)
- 2048→512→256→5
- **25M parameters** (vs 11M for ResNet18)

**Training:**
- Epochs: 10
- Very unstable! Val R² ranged from -5.37 to +0.55

**Performance:**
- Validation R²: **+0.5529**
- **Worse than ResNet18!**

**Why?** Dataset too small for ResNet50 (285 images can't train 25M parameters).

**Can submit?** ✅ Yes (but not recommended)

---

## Key Insights from All Models

### 1. Complexity Doesn't Always Help

| Complexity | Model | Val R² |
|------------|-------|--------|
| ⭐⭐⭐ High | Model 4b (30 epochs, auxiliary) | +0.6852 |
| ⭐⭐ Medium | Model 1 (10 epochs, simple) | +0.6352 |
| ⭐⭐⭐⭐ Very High | Model 5 (ResNet50) | +0.5529 |

**Lesson:** More complexity helps on **validation** but may hurt on **test** (overfitting).

### 2. ColorJitter Hurts

| Model | ColorJitter? | Val R² | Difference |
|-------|--------------|--------|------------|
| Model 1 | ❌ No | +0.6352 | Baseline |
| Model 2 | ✅ Yes | +0.5584 | **-0.0768** |

**Lesson:** Green/brown colors are critical features for biomass prediction!

### 3. Training Duration Matters

| Model | Epochs | Val R² | Kaggle R² |
|-------|--------|--------|-----------|
| Model 4b | 30 | +0.6852 | +0.51 (gap: -0.18) |
| Model 1 | 10 | +0.6352 | Testing next |

**Lesson:** More epochs improves validation but may hurt generalization.

### 4. Multimodal is Great (But Not Submittable)

| Model | Type | Val R² | Can Submit? |
|-------|------|--------|-------------|
| Model 3/4 Teacher | Multimodal | +0.66 | ❌ No |
| Model 4b | Auxiliary (bakes tabular into image) | +0.69 | ✅ Yes |
| Model 1 | Image-only | +0.64 | ✅ Yes |

**Lesson:** Tabular features help, but need clever ways to use them (like auxiliary pretraining).

### 5. Model Size Doesn't Matter Much

| Model | Parameters | Val R² |
|-------|------------|--------|
| Model 1-4b | ~11M | +0.63 to +0.69 |
| Model 5 | ~25M | +0.55 |

**Lesson:** Dataset is too small for very large models.

---

## What to Do Next

### Step 1: Submit Model 1 ⭐ PRIORITY!

**Why:**
- Test if simpler model generalizes better
- If it scores > 0.51, confirms Model 4b overfit
- Fastest way to understand the problem

**How:**
1. Open `15_model1_submission.ipynb`
2. Run all cells
3. Download `submission.csv`
4. Submit to Kaggle

### Step 2: Based on Results

**If Model 1 > 0.55:**
- ✅ Use Model 1!
- Or train Model 1 for 20 epochs
- Or ensemble Model 1 + Model 4b C

**If 0.52 < Model 1 < 0.55:**
- Retrain Model 4b with early stopping (15 epochs)
- Use full dataset normalization
- Higher dropout (0.3)

**If Model 1 ≈ 0.51:**
- Ensemble multiple models
- More data augmentation
- Consider external data

**If Model 1 < 0.48:**
- Debug normalization
- Check data preprocessing
- Investigate test distribution

---

## Files You Have

### Notebooks
- ✅ `11_model_comparison.ipynb` - Trained Models 1-5 (10 epochs each)
- ✅ `13_model4b_final_training.ipynb` - Trained Model 4b variants (30 epochs)
- ✅ `14_kaggle_submission.ipynb` - Model 4b submission (already submitted, R²=0.51)
- ✅ `15_model1_submission.ipynb` - Model 1 submission (**READY TO SUBMIT!**)

### Checkpoints (All 43-44 MB each)
- ✅ `Model_1_Simple_best.pth` (+0.6352)
- ✅ `Model_2_ColorJitter_best.pth` (+0.5584)
- ✅ `Model_3_Multimodal_best.pth` (+0.6055)
- ✅ `Model_4_Teacher_best.pth` (+0.6642)
- ✅ `Model_4_Student_best.pth` (+0.5553)
- ✅ `Model_5_ResNet50_best.pth` (+0.5529)
- ✅ `model4b_A_Baseline_phase2_best.pth` (+0.6852)
- ✅ `model4b_B_HigherLR_phase2_best.pth` (+0.6641)
- ✅ `model4b_C_LargerHead_phase2_best.pth` (+0.6711)

---

**Bottom line: Your 0.51 score is decent but you can likely do better! Submit Model 1 next to find out if overfitting is the issue.** 🚀
