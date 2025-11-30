# Kaggle Submission Guide - Universal Features K-Fold Ensemble

## Quick Start

You have **2 files ready** for Kaggle submission:

1. **Notebook**: `25_universal_kfold_kaggle_submission.ipynb`
2. **Model Checkpoints**: `universal_Fold{1-5}_best.pth` (5 files, ~45MB each)

---

## Step 1: Upload Model Checkpoints as Kaggle Dataset

### Create New Dataset

1. Go to [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click **"New Dataset"**
3. Upload **5 checkpoint files**:
   - `universal_Fold1_best.pth`
   - `universal_Fold2_best.pth`
   - `universal_Fold3_best.pth`
   - `universal_Fold4_best.pth`
   - `universal_Fold5_best.pth`

### Dataset Settings

- **Title**: `CSIRO Biomass - Universal K-Fold Models`
- **Subtitle**: `5-Fold CV models trained with universal features + species`
- **Description**:
  ```
  ResNet18 models trained with K-Fold cross-validation using universal features.

  Key Features:
  - Universal: NDVI, Height, Daylength, Season, Species (15 classes)
  - Removed: State, Weather (location-specific)
  - 5 folds for ensemble averaging
  - Each model ~45MB, ~11M parameters

  Expected Kaggle score: 0.52-0.54
  (Previous K-Fold with location features: 0.50)
  ```
- **Visibility**: Private (or Public if you want to share)

4. Click **"Create"**
5. **Note the dataset name** (e.g., `username/csiro-biomass-universal-kfold`)

---

## Step 2: Create Kaggle Notebook

### Upload Notebook

1. Go to competition: [CSIRO Biomass Prediction](https://www.kaggle.com/competitions/csiro-biomass-prediction)
2. Click **"Code"** tab
3. Click **"New Notebook"**
4. Delete default content
5. Click **"File"** → **"Import Notebook"**
6. Upload: `25_universal_kfold_kaggle_submission.ipynb`

### Add Data Sources

After uploading notebook:

1. Click **"Add Data"** button (right sidebar)
2. Add **Competition Data**:
   - Search: `csiro-biomass`
   - Click **"Add"**
3. Add **Your Dataset**:
   - Search: `csiro-biomass-universal-kfold` (or your dataset name)
   - Click **"Add"**

### Verify Paths

The notebook should auto-detect paths:
- Test data: `/kaggle/input/csiro-biomass/test.csv`
- Models: `../input/YOUR-DATASET-NAME/universal_Fold{1-5}_best.pth`

If paths don't match, update **Cell 4** checkpoint paths:
```python
checkpoint_paths = [
    f'../input/YOUR-ACTUAL-DATASET-NAME/{checkpoint_name}',  # Update this line
    ...
]
```

---

## Step 3: Run Notebook

### Settings

1. **Accelerator**: GPU T4 x2 (or any GPU)
   - Path: Settings → Accelerator
   - Universal models work with CPU too, but GPU is faster
2. **Internet**: OFF (notebook works offline!)
   - Path: Settings → Internet
   - Models don't need ImageNet downloads

### Run

1. Click **"Run All"** or **"Save Version"** → **"Run All"**
2. Wait ~5-10 minutes:
   - Load 5 models: ~30 seconds
   - Generate predictions: ~3-5 minutes
   - Create submission: ~10 seconds
3. Check output:
   - Look for: `✅ SUBMISSION FILE CREATED: submission.csv`
   - Verify: No errors in Cell 4 (model loading)

---

## Step 4: Submit to Competition

### Download Submission

1. After notebook finishes, check **Output** section
2. Find `submission.csv` in files list
3. Click to download

### Submit

1. Go to competition: [Submit Predictions](https://www.kaggle.com/competitions/csiro-biomass-prediction/submit)
2. Click **"Submit Predictions"**
3. Upload `submission.csv`
4. Add description:
   ```
   Universal Features K-Fold Ensemble (5 folds)
   - Architecture: ResNet18 + Auxiliary Pretraining
   - Features: NDVI, Height, Season, Daylength, Species
   - Removed: State, Weather (location-specific)
   - Expected: 0.52-0.54
   ```
5. Click **"Submit"**

---

## Expected Results

### Comparison

| Approach | Validation R² | Kaggle R² | Notes |
|----------|--------------|-----------|-------|
| Baseline (single ResNet18) | +0.69 | **+0.51** | Single 80/20 split |
| K-Fold with location features | +0.90 | **+0.50** | Learned location bias! |
| **Universal K-Fold** (this) | ~+0.68 | **+0.52-0.54** | Should generalize! |

### Why This Should Work

✅ **Universal features generalize** to new locations
✅ **Species is universal** - Ryegrass similar everywhere
✅ **No State/Weather** - no location bias
✅ **Ensemble effect** - 5 models reduce overfitting

### If Score is Good (≥0.52)

Confirms location bias was the problem! Next improvements:
- Try ResNet50 or EfficientNet backbone
- Tune data augmentation
- Optimize ensemble weights

### If Score is Still Low (<0.52)

May indicate other distribution shifts. Try:
- Test-time augmentation (TTA)
- Different CV split strategy
- Analyze which target is worst (check per-target R²)

---

## Troubleshooting

### Error: "Checkpoint not found"

**Solution**: Update checkpoint paths in Cell 4
```python
checkpoint_paths = [
    f'../input/YOUR-DATASET-NAME/{checkpoint_name}',  # Must match your dataset name
    ...
]
```

### Error: "Test images not found"

**Solution**: Verify competition data was added
1. Check: Right sidebar → "Data" section
2. Should see: `csiro-biomass-prediction`
3. If missing: Click "Add Data" → Add competition

### Error: "CUDA out of memory"

**Solution 1**: Reduce batch size in Cell 2
```python
BATCH_SIZE = 8  # Changed from 16
```

**Solution 2**: Use CPU instead
- Remove GPU accelerator in notebook settings
- Inference will be slower (~15 min instead of 5 min)

### Submission Format Error

Check Cell 8 output:
- ✓ Correct columns: `['sample_id', 'target']`
- ✓ No NaN values
- ✓ No negative values
- ✓ Correct number of rows (num_test_images × 5)

---

## Files Summary

### Local Files (In Your Project)

```
/Users/tim/Code/Tim/csiro-biomass/
├── 24_kfold_universal_features.ipynb     # Training notebook
├── 25_universal_kfold_kaggle_submission.ipynb  # Submission notebook (THIS ONE)
├── universal_Fold1_best.pth              # Model 1 (~45MB)
├── universal_Fold2_best.pth              # Model 2 (~45MB)
├── universal_Fold3_best.pth              # Model 3 (~45MB)
├── universal_Fold4_best.pth              # Model 4 (~45MB)
├── universal_Fold5_best.pth              # Model 5 (~45MB)
└── competition/
    └── test.csv                          # Test data (for local testing)
```

### Kaggle Files Structure

After uploading, your Kaggle environment will look like:
```
/kaggle/
├── input/
│   ├── csiro-biomass/
│   │   ├── test.csv
│   │   └── test/ (images)
│   └── csiro-biomass-universal-kfold/  # Your dataset
│       ├── universal_Fold1_best.pth
│       ├── universal_Fold2_best.pth
│       ├── universal_Fold3_best.pth
│       ├── universal_Fold4_best.pth
│       └── universal_Fold5_best.pth
└── working/
    └── submission.csv  # Generated output
```

---

## Quick Checklist

Before submitting:

- [ ] Uploaded 5 checkpoint files as Kaggle dataset
- [ ] Created new Kaggle notebook
- [ ] Added competition data source
- [ ] Added model checkpoint dataset source
- [ ] Updated checkpoint paths in Cell 4 (if needed)
- [ ] Set accelerator to GPU (optional but faster)
- [ ] Set internet to OFF (works offline)
- [ ] Ran notebook successfully
- [ ] Downloaded submission.csv
- [ ] Submitted to competition

After submitting:

- [ ] Check Kaggle score (should be 0.52-0.54)
- [ ] Compare with previous attempts (baseline 0.51, prev K-Fold 0.50)
- [ ] Note which approach worked best
- [ ] Decide on next improvements if needed

---

## Contact & Help

If you encounter issues:

1. **Check notebook output** - Look for error messages in Cell 4 (model loading) or Cell 7 (predictions)
2. **Verify data paths** - Use Cell 5 output to confirm test.csv was found
3. **Check file sizes** - submission.csv should be ~5-10KB for typical test set
4. **Kaggle forums** - Search for similar issues in competition discussion

---

Good luck! 🚀

Expected improvement: **+0.01 to +0.03** over previous K-Fold (0.50)

Key insight: **Universal features generalize better to unseen locations!**
