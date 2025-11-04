# Overnight Training Guide

## ✅ Notebook is Ready!

All code has been validated for syntax errors and is ready for overnight training.

## What Changed

### 1. **Extended Training with Early Stopping**
- **Baseline**: Up to 40 epochs (will stop early if plateaus)
- **Teacher**: Up to 50 epochs (will stop early if plateaus)
- **Student**: 50 epochs (distillation needs full training)
- **Auxiliary**: 50 epochs (multi-task needs full training)
- **Early stopping patience**: 8 epochs without improvement

### 2. **R² Tracking During Training**
Every epoch now shows:
```
Epoch 15/50
  Train Loss: 0.4523 | Val Loss: 0.4198 | Val R²: 0.5234
  💾 Saved best model
  ✓ Improvement: 0.000124
```

### 3. **Early Stopping Messages**
When a model plateaus:
```
  No improvement for 8/8 epochs
  🛑 Early stopping at epoch 28
     Best R² = 0.5389 at epoch 20

✓ Training complete! Best R² = 0.5389
```

## How to Run Overnight

### Step 1: Prevent Mac from Sleeping
Open Terminal and run:
```bash
caffeinate -dims &
```

This prevents your Mac from sleeping even with the lid closed. **Keep plugged into power!**

To check it's running:
```bash
ps aux | grep caffeinate
```

To stop it later:
```bash
killall caffeinate
```

### Step 2: Start Training
1. Open the notebook in VSCode
2. **Verify Cell 4 shows**: `DEBUG_MODE = False` ✓
3. **Run All Cells** (Cmd+Shift+P → "Run All")
4. Training will start automatically

### Step 3: Monitor Progress (Optional)
You can monitor remotely via:
- **Screen Sharing**: Enable in System Preferences → Sharing
- **Check from another device**: Connect via Screen Sharing app

Or just check in the morning!

## Expected Timeline

Based on early stopping at ~25-35 epochs per model:

| Model | Expected Epochs | Time |
|-------|----------------|------|
| Baseline | 20-30 epochs | 30-45 min |
| Teacher | 25-35 epochs | 45-60 min |
| Student | 50 epochs (full) | 90-120 min |
| Auxiliary | 50 epochs (full) | 90-120 min |
| **Total** | | **4-5 hours** |

## What to Expect

### Initial Epochs (1-5):
- R² will be **negative** (~-2.0 to -1.0)
- This is normal! Models are learning from random initialization

### Early Training (5-15):
- R² should become **positive** (0.0 to 0.3)
- Loss decreasing steadily
- You'll see "💾 Saved best model" frequently

### Mid Training (15-25):
- R² reaching **0.4-0.6**
- Improvements getting smaller
- Learning rate may decrease (ReduceLROnPlateau)

### Late Training (25+):
- R² plateaus around **0.5-0.7**
- Early stopping may trigger for Baseline/Teacher
- "No improvement for X/8 epochs" messages

### Final Results (Expected):
```
Competition Scores (Weighted R²):
  • Baseline:  0.45 - 0.55
  • Teacher:   0.60 - 0.75  (best, has all data)
  • Student:   0.50 - 0.65  (learned from teacher)
  • Auxiliary: 0.50 - 0.65  (learned environmental features)
```

## Troubleshooting

### If Mac Sleeps Anyway:
1. Check System Preferences → Energy Saver
2. Set "Prevent computer from sleeping automatically" to ON
3. Set "Display sleep" to Never (while plugged in)
4. Run `caffeinate -dims` again

### If Training Fails:
All models save checkpoints (`*_best.pth`). You can:
1. Check which models completed
2. Skip completed training cells
3. Run evaluation cells to see results
4. Restart failed models from scratch

### If You Need to Check Progress:
```bash
# In another terminal, monitor output
tail -f ~/Library/Logs/com.microsoft.VSCode/*.log
```

Or just reopen the notebook - outputs are cached!

## After Training Completes

### Morning Checklist:
1. ✓ Check the final comparison table (Cell 37)
2. ✓ Look at the R² visualization (Cell 38)
3. ✓ Note which model won
4. ✓ Check how many epochs each model actually trained
5. ✓ Stop caffeinate: `killall caffeinate`

### Next Steps:
- Review learning curves to see training dynamics
- If results are good, generate test predictions
- If results are poor, investigate what went wrong
- Consider hyperparameter tuning

## Safety Features

✅ **Checkpointing**: Best model always saved
✅ **Early stopping**: Won't waste time on plateaued models
✅ **Cached outputs**: VSCode preserves everything
✅ **Resume capable**: Can skip completed cells
✅ **Progress tracking**: R² printed every epoch

## Questions?

- **"Can I close the lid?"**: Yes, with `caffeinate -dims` running
- **"What if it stops?"**: Check System Preferences energy settings
- **"How do I know it's working?"**: Check notebook outputs or use Screen Sharing
- **"What if power fails?"**: Load checkpoints and restart from failed model
- **"Should I watch it?"**: No need! Check in the morning

Good luck! 🚀
