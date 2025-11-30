# Kaggle Submission Guide - K-Fold Ensemble

## 📦 Step 1: Upload Model Checkpoints as Kaggle Dataset

You need to upload the 5 trained fold models to Kaggle as a dataset.

### Files to Upload:
```
model4b_Fold1_phase2_best.pth
model4b_Fold2_phase2_best.pth
model4b_Fold3_phase2_best.pth
model4b_Fold4_phase2_best.pth
model4b_Fold5_phase2_best.pth
```

### How to Upload:

1. **Go to Kaggle Datasets**: https://www.kaggle.com/datasets
2. **Click "New Dataset"**
3. **Upload the 5 .pth files** (you can drag & drop all 5 at once)
4. **Configure the dataset**:
   - **Title**: `CSIRO Biomass K-Fold Models` (or any name you prefer)
   - **Slug**: `csiro-biomass-kfold-models` (note this - you'll need it!)
   - **Description**:
     ```
     5-Fold Cross-Validation ensemble models for CSIRO Biomass Prediction

     - Architecture: ResNet18 + Auxiliary Pretraining
     - Ensemble Val R²: +0.9007
     - Individual fold R² (mean): +0.6532
     - Expected Kaggle: 0.53-0.55
     ```
   - **Visibility**: Private (or Public if you want to share)
5. **Click "Create"**

### File Locations:
The models should be in your current directory:
```bash
ls -lh model4b_Fold*_phase2_best.pth
```

If they're not there, they might be in `experiments/option4_kfold_cv/`

---

## 📓 Step 2: Upload Submission Notebook to Kaggle

### File to Upload:
```
23_kfold_ensemble_kaggle_submission.ipynb
```

### How to Upload:

1. **Go to the Competition**: https://www.kaggle.com/competitions/csiro-biomass-prediction
2. **Click "Code" tab**
3. **Click "New Notebook"**
4. **Click the three dots menu (⋮)** in the notebook editor
5. **Select "Import Notebook"**
6. **Upload `23_kfold_ensemble_kaggle_submission.ipynb`**

---

## 🔗 Step 3: Add Model Dataset as Input

Once your notebook is uploaded to Kaggle:

1. **Click "Add Data" button** (right sidebar in notebook editor)
2. **Search for your dataset**: `csiro-biomass-kfold-models` (or the slug you used)
3. **Click the "+" button** to add it as input
4. **Verify the path**: The notebook will look for models at:
   ```
   ../input/csiro-biomass-kfold-models/model4b_Fold1_phase2_best.pth
   ../input/csiro-biomass-kfold-models/model4b_Fold2_phase2_best.pth
   ...
   ```

### If Your Dataset Has a Different Name:

Edit Cell 4 in the notebook and update this line:
```python
f'../input/YOUR-DATASET-SLUG-HERE/{checkpoint_name}',  # Kaggle input
```

Replace `YOUR-DATASET-SLUG-HERE` with your actual dataset slug.

---

## ▶️ Step 4: Run the Notebook on Kaggle

1. **Check Settings**:
   - **Accelerator**: GPU (optional, but faster) or CPU (works fine)
   - **Internet**: OFF (not needed for inference)
   - **Persistence**: Files only

2. **Click "Save Version"** → **"Save & Run All"**

3. **Wait for completion** (~5-10 minutes with GPU, ~15-20 minutes with CPU)

4. **Check for success**: Look for:
   ```
   ✅ SUBMISSION FILE CREATED: submission.csv
   ```

---

## 📤 Step 5: Submit to Competition

1. **After notebook finishes running**, click "Submit" button (top right)
2. **Or**: Download `submission.csv` from notebook output and submit via competition page

---

## ✅ Expected Results

| Metric | Value |
|--------|-------|
| **Ensemble Val R²** | +0.9007 |
| **Baseline Kaggle** | +0.51 |
| **Expected Kaggle** | **0.53-0.55** |
| **Expected Improvement** | **+0.02 to +0.04** |

---

## 🔍 Troubleshooting

### Issue: "Could not find model checkpoint for Fold X"

**Solution**:
- Make sure all 5 .pth files are uploaded to your Kaggle dataset
- Check that the dataset is added as input to the notebook
- Verify the dataset slug matches in Cell 4

### Issue: "Could not find test.csv"

**Solution**:
- Make sure you're running the notebook in the competition environment
- Add the competition data as input: Search for "csiro-biomass-prediction" in Add Data

### Issue: Notebook times out

**Solution**:
- Enable GPU accelerator (Settings → Accelerator → GPU T4 x2)
- This reduces inference time from ~15 min to ~5 min

### Issue: Memory error

**Solution**:
- Reduce BATCH_SIZE from 16 to 8 in Cell 2
- Or use GPU accelerator which has more memory

---

## 📊 Comparison with Previous Submissions

| Model | Val R² | Kaggle R² | Notes |
|-------|--------|-----------|-------|
| Model 4b (Baseline) | 0.6852 | 0.51 | Single model, -0.175 gap |
| **K-Fold Ensemble** | **0.9007** | **0.53-0.55 (expected)** | **5 models averaged** |

---

## 💡 Tips

1. **Test locally first**: Run the notebook locally with your checkpoint files to verify it works
2. **Check file sizes**: Each .pth file should be ~40-50 MB
3. **Use GPU**: Much faster inference (~3x speedup)
4. **Monitor output**: Watch for the "✅ Successfully loaded all 5 fold models!" message

---

## 🎯 Success Criteria

✅ All 5 model checkpoints loaded successfully
✅ submission.csv created with correct format
✅ No NaN, infinite, or negative values
✅ Correct number of rows (num_images × 5 targets)
✅ Kaggle score improves from baseline 0.51

---

## 📁 File Checklist

Before starting, make sure you have:

- [ ] `model4b_Fold1_phase2_best.pth` (~40-50 MB)
- [ ] `model4b_Fold2_phase2_best.pth` (~40-50 MB)
- [ ] `model4b_Fold3_phase2_best.pth` (~40-50 MB)
- [ ] `model4b_Fold4_phase2_best.pth` (~40-50 MB)
- [ ] `model4b_Fold5_phase2_best.pth` (~40-50 MB)
- [ ] `23_kfold_ensemble_kaggle_submission.ipynb`

Total size: ~200-250 MB for all 5 models

---

Good luck! 🍀 Let me know what Kaggle score you get!
