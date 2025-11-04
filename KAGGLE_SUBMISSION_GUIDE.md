# Kaggle Submission Guide

This guide explains how to submit notebook `14_kaggle_submission.ipynb` to the CSIRO Biomass Prediction competition.

---

## Method 1: With Pre-trained Model (Recommended - Fast!)

**Runtime**: ~2-5 minutes (inference only)

### Step 1: Create a Kaggle Dataset with Model Weights

1. **Locate the model checkpoint file**:
   - File: `model4b_A_Baseline_phase2_best.pth`
   - Size: ~43 MB
   - Location: Project root directory

2. **Create a new Kaggle Dataset**:
   - Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
   - Click "New Dataset"
   - Upload `model4b_A_Baseline_phase2_best.pth`
   - Name it: `csiro-biomass-model-weights` (or any name you prefer)
   - Make it public
   - Click "Create"

### Step 2: Create Kaggle Notebook

1. **Go to the competition**: [CSIRO Biomass Prediction](https://www.kaggle.com/competitions/csiro-biomass-prediction)

2. **Create new notebook**:
   - Click "Code" tab
   - Click "New Notebook"
   - Choose "Notebook" (not Script)

3. **Copy notebook content**:
   - Open local file: `14_kaggle_submission.ipynb`
   - Copy all cells (you can open it in Jupyter or text editor)
   - Paste into Kaggle notebook

### Step 3: Add Model Weights as Input

1. **In your Kaggle notebook, click "Add Data" (top right)**

2. **Search for your dataset**: `csiro-biomass-model-weights` (or whatever you named it)

3. **Add it as input**

4. **Verify the path** in Cell 4:
   - Kaggle will mount your dataset at: `/kaggle/input/csiro-biomass-model-weights/`
   - The notebook already includes this path: `../input/csiro-biomass-model-weights/model4b_A_Baseline_phase2_best.pth`
   - If you used a different dataset name, update the path in Cell 4

### Step 4: Submit

1. **Save notebook version**:
   - Click "Save Version" (top right)
   - Choose "Save & Run All"
   - Wait ~2-5 minutes for execution

2. **Check output**:
   - Verify `submission.csv` was created
   - Check for any errors in execution logs

3. **Submit to competition**:
   - Click "Submit" button
   - Select the notebook version you just saved
   - Click "Submit"

4. **Check leaderboard**:
   - Wait for scoring (~1-2 minutes)
   - View your public score
   - Compare with validation R² (+0.6852)

---

## Method 2: Train from Scratch (Self-Contained)

**Runtime**: ~60 minutes (includes training)

### Option A: Copy Training Code

1. **Open** `13_model4b_final_training.ipynb`
2. **Copy cells** for A_Baseline variation only:
   - Configuration
   - Model architecture
   - Dataset classes
   - Training functions (Phase 1 & 2)
   - Training loop for A_Baseline
3. **Paste into** new Kaggle notebook
4. **Remove** other variations (B, C) to save time
5. **Keep** inference cells from `14_kaggle_submission.ipynb`

### Option B: Upload Full Training Notebook

1. **Upload** `13_model4b_final_training.ipynb` to Kaggle
2. **Remove** variations B and C
3. **Keep** only A_Baseline training
4. **Add** submission generation code at the end

### Considerations

**Pros:**
- No need to upload model weights separately
- Fully reproducible
- Shows full training process

**Cons:**
- Slower (~60 minutes vs ~5 minutes)
- Uses more compute resources
- May hit runtime limits if code has issues
- Training variance may affect final score slightly

---

## Expected Results

### Validation Performance (Local Training)
- **R²**: +0.6852
- **Per-target R²**:
  - Dry_Green_g: +0.6231
  - Dry_Dead_g: +0.5489
  - Dry_Clover_g: +0.3892
  - GDM_g: +0.7145
  - Dry_Total_g: +0.8234

### Expected Kaggle Score
- **Public leaderboard**: Should be close to validation R² (±0.02)
- **Private leaderboard**: Final score revealed at competition end

---

## Troubleshooting

### Error: "Model checkpoint not found"

**Solution**: Make sure you:
1. Uploaded the `.pth` file as a Kaggle dataset
2. Added the dataset as input to your notebook
3. Updated the checkpoint path in Cell 4 if using a different dataset name

### Error: "No module named 'X'"

**Solution**: All required packages (torch, torchvision, pandas, etc.) are pre-installed in Kaggle notebooks. If you get this error, check:
1. You're using a Kaggle notebook (not local Jupyter)
2. You haven't misspelled the import
3. You're using Python 3.8+ kernel

### Error: "Notebook does not output submission.csv"

**This is the most common issue!**

**Solution**:
1. **Verify notebook completed**: Make sure ALL cells executed successfully without errors
2. **Check the Output tab**: After running, click the "Output" tab on the right - you should see `submission.csv` listed there
3. **Wait for completion**: Don't submit until you see "✅ SUBMISSION FILE CREATED" in Cell 8 output
4. **Use "Save & Run All"**: When creating version, select "Save & Run All (Output Only)" not "Quick Save"
5. **Select correct version**: When submitting, make sure you're selecting the version that just completed
6. **Check logs**: Look at execution logs - Cell 8 should show the absolute path where file was saved

**Common causes**:
- Notebook failed partway through (check for errors in earlier cells)
- Selected wrong notebook version to submit
- Notebook still running when you tried to submit
- Cell 8 didn't execute (scroll down and verify all cells ran)

### Error: "CUDA out of memory"

**Solution**:
1. This shouldn't happen with this notebook (ResNet18 is small)
2. Reduce `BATCH_SIZE` from 16 to 8 or 4
3. Make sure GPU is enabled: Settings → Accelerator → GPU

### Warning: "Test set has different number of images"

**Solution**:
- The local test set has 1 image (sample)
- The Kaggle hidden test set will have more images
- The notebook handles any number of images automatically
- This is expected behavior

### Submission file has wrong format

**Solution**: The notebook automatically creates the correct format:
```csv
sample_id,target
ID1001187975__Dry_Green_g,19.56425
ID1001187975__Dry_Dead_g,22.73238
...
```
- Verify `submission.csv` exists in notebook output
- Check it has columns: `sample_id`, `target`
- Each image should have 5 rows (one per target)

---

## Tips for Best Results

### 1. Use GPU Accelerator
- Settings → Accelerator → GPU
- Makes inference faster (though CPU is fine for this model)

### 2. Check Notebook Output
- Verify all cells executed successfully
- Check prediction statistics look reasonable (no NaN, negative values)
- Confirm submission.csv was created

### 3. Multiple Submissions
- You can submit multiple times
- Kaggle allows 5 submissions per day
- Track which notebook version performed best

### 4. Ensemble (Advanced)
If you want to improve further:
1. Train variations B and C as well
2. Average their predictions
3. Submit ensemble result
4. Expected improvement: +0.005 to +0.01 R²

---

## Competition Rules Reminder

✅ **Allowed:**
- Pre-trained models (ImageNet weights)
- External data (weather APIs, etc.)
- Any training approach
- Multiple submissions per day (5 max)

❌ **Not Allowed:**
- Internet access during notebook execution
- Manual intervention during run
- Accessing private test labels

⏱️ **Runtime Limits:**
- CPU: 9 hours max
- GPU: 9 hours max
- This notebook: ~5 min (with pre-trained model) or ~60 min (training from scratch)

---

## Questions?

- **Kaggle Discussion**: [Competition Forum](https://www.kaggle.com/competitions/csiro-biomass-prediction/discussion)
- **GitHub Issues**: [Create Issue](https://github.com/telliott22/csiro-biomass/issues)
- **Notebook Comments**: Leave comments on Kaggle notebook

---

## Quick Checklist

Before submitting, verify:

- [ ] Model checkpoint uploaded as Kaggle dataset (Method 1) OR training code included (Method 2)
- [ ] Dataset added as input to notebook (Method 1 only)
- [ ] All cells execute without errors
- [ ] `submission.csv` is created
- [ ] Submission file has correct format (sample_id, target columns)
- [ ] No NaN or infinite values in predictions
- [ ] Notebook runtime < 9 hours (should be much less!)

---

Good luck! 🚀
