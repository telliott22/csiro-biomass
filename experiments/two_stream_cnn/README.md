# Two-Stream CNN Experiment

## Quick Start

```bash
cd experiments/two_stream_cnn
jupyter notebook 28_two_stream_test.ipynb
```

Run all cells to train the two-stream model (~2 hours).

## Files

- **28_two_stream_test.ipynb** - Training notebook
- **TWO_STREAM_EXPLANATION.md** - Architecture documentation
- **README.md** - This file

## Expected Output

After training:
- `two_stream_best.pth` - Best model checkpoint
- `training_history.png` - Loss/R² plots

## Architecture

**Stream 1:** RGB image → ResNet18 → 512 features
**Stream 2:** NDVI image → ResNet18 → 512 features
**Fusion:** Concatenate → MLP → 5 biomass predictions

## Expected Performance

**Optimistic:** Val R² ≥ 0.69, Kaggle ≥ 0.52
**Realistic:** Val R² ≈ 0.65-0.68, Kaggle ≈ 0.51-0.52
**Pessimistic:** Val R² < 0.65, Kaggle < 0.51

## Next Steps

- If Val R² ≥ 0.69: Create Kaggle submission (29_two_stream_submission.ipynb)
- If Val R² < 0.69: Try longer training or different fusion strategy
