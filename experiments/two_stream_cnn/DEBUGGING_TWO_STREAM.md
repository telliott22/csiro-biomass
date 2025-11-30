# Debugging Two-Stream CNN Poor Performance

## Likely Issues

### Issue #1: NDVI Stream is Useless (Most Likely!)

**Problem:** NDVI "image" is spatially constant (all pixels = same value)

```python
ndvi_img = torch.full((224, 224), 0.75)  # Every pixel is 0.75
```

**Why this is bad:**
- CNNs learn spatial patterns
- No spatial variation → nothing to learn
- Conv layers see identical values everywhere
- NDVI stream learns nothing useful
- Might as well be a single scalar input!

**Evidence this is the issue:**
- If NDVI stream is useless, model reduces to single RGB stream
- Performance would match or be worse than baseline (0.51)
- Validation R² would be around 0.65-0.69 (similar to baseline)

---

### Issue #2: Two Streams Interfere During Training

**Problem:** Training two streams from scratch simultaneously

**Why this is bad:**
- Both streams compete for gradient updates
- Neither stream converges well
- Random initialization → slow learning
- May converge to local minimum

**Evidence:**
- Training loss decreases slowly
- Validation R² plateaus early
- One stream dominates, other contributes little

---

### Issue #3: NDVI Normalization is Wrong

**Problem:** NDVI normalized using train set statistics

```python
ndvi_normalized = (ndvi_value - ndvi_mean) / ndvi_std
```

**Why this might be bad:**
- NDVI typically ranges [0, 1] or [-1, 1]
- Standard normalization might push values far from useful range
- NDVI stream sees very different distribution than RGB stream

---

### Issue #4: Model is Too Large / Underfitting

**Problem:** 22M parameters with only ~285 training images

**Why this is bad:**
- Severe overfitting (but you said performance is poor, so probably not this)
- OR underfitting if not enough epochs
- Need more regularization

---

## Diagnostic Questions

To pinpoint the issue, check:

1. **What's the validation R²?**
   - < 0.60: Major problem
   - 0.60-0.65: Significant issue
   - 0.65-0.69: Minor issue (maybe just noise)
   - ≥ 0.69: Actually okay!

2. **Does training loss decrease?**
   - Stuck high: Not learning at all
   - Decreases slowly: Training issue
   - Decreases fast then plateaus: Capacity issue

3. **What's the training R² vs validation R²?**
   - Train high, val low: Overfitting
   - Both low: Not learning properly
   - Both high: Actually working!

---

## Likely Root Cause: Constant NDVI Image

The most probable issue is **Issue #1**: The NDVI stream sees no spatial variation.

### Why Constant NDVI is Problematic

**What CNN needs:** Spatial patterns to extract
```
Good image (real photo):
[120, 125, 130, 122, ...]  ← Variation!
[118, 135, 128, 140, ...]
[142, 150, 138, 145, ...]
```

**What we're giving NDVI stream:**
```
NDVI "image":
[0.75, 0.75, 0.75, 0.75, ...]  ← No variation!
[0.75, 0.75, 0.75, 0.75, ...]
[0.75, 0.75, 0.75, 0.75, ...]
```

**Result:** NDVI stream learns nothing useful!

---

## Solutions to Try

### Solution 1: Use NDVI as Scalar Feature (Best)

**Abandon two-stream for NDVI. Use it properly:**

```python
class ImprovedModel(nn.Module):
    def __init__(self):
        # Single RGB stream
        self.rgb_stream = ResNet18()  # → 512 features

        # NDVI as scalar feature
        self.ndvi_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )  # → 128 features

        # Fusion
        self.fusion = nn.Linear(512 + 128, 5)  # Biomass predictions
```

**Forward pass:**
```python
def forward(self, rgb_img, ndvi_scalar):
    rgb_features = self.rgb_stream(rgb_img)        # [batch, 512]
    ndvi_features = self.ndvi_encoder(ndvi_scalar)  # [batch, 128]
    combined = torch.cat([rgb_features, ndvi_features], dim=1)
    return self.fusion(combined)
```

**Why this works:**
- RGB stream learns visual patterns
- NDVI scalar properly encoded as feature
- No fake spatial structure
- Simpler, more honest architecture

---

### Solution 2: Create Multi-Channel "Feature Image"

**Make second stream actually useful:**

```python
def create_feature_image(row):
    # Create 3-channel feature image
    ndvi_channel = torch.full((224, 224), row['Pre_GSHH_NDVI'])
    height_channel = torch.full((224, 224), row['Height_Ave_cm'] / 100)  # Normalize
    season_channel = torch.full((224, 224), row['season'] / 4)  # 0-1 range

    feature_img = torch.stack([ndvi_channel, height_channel, season_channel])
    return feature_img  # [3, 224, 224]
```

**Why this might work:**
- Still no spatial variation, but more features
- Second stream learns "if NDVI=X and Height=Y → expect Z biomass"
- More honest than pretending NDVI is spatial

**But still limited!** CNNs want spatial patterns.

---

### Solution 3: Skip Two-Stream, Use Auxiliary Pretraining (Already Tried)

This is essentially what we did before! Baseline with auxiliary tasks.

The two-stream idea was to separate streams, but if one stream is useless...

---

### Solution 4: Get Actual NDVI Rasters

**If you had spatial NDVI data:**

```python
# If you had NDVI raster per image
ndvi_raster = load_ndvi_raster(image_id)  # [224, 224] with spatial variation!

# Then second stream would learn spatial vegetation patterns
```

**Problem:** We don't have per-image NDVI rasters, only scalar values.

---

## Recommended Action

### Immediate: Check What's Happening

Add diagnostic cell to notebook:

```python
# After training, check feature magnitudes
model.eval()
sample_batch = next(iter(val_loader))
rgb_imgs = sample_batch['rgb_image'].to(device)
ndvi_imgs = sample_batch['ndvi_image'].to(device)

with torch.no_grad():
    rgb_features = model.rgb_stream(rgb_imgs).flatten(1)
    ndvi_features = model.ndvi_stream(ndvi_imgs).flatten(1)

    print(f"RGB features:")
    print(f"  Mean: {rgb_features.mean():.4f}")
    print(f"  Std: {rgb_features.std():.4f}")
    print(f"  Range: [{rgb_features.min():.4f}, {rgb_features.max():.4f}]")

    print(f"\nNDVI features:")
    print(f"  Mean: {ndvi_features.mean():.4f}")
    print(f"  Std: {ndvi_features.std():.4f}")
    print(f"  Range: [{ndvi_features.min():.4f}, {ndvi_features.max():.4f}]")

    # Check if NDVI stream collapsed to zeros
    if ndvi_features.std() < 0.01:
        print("\n❌ NDVI stream is dead! (Near-zero variation)")
    else:
        print("\n✓ NDVI stream is active")
```

---

### Next Steps Based on Diagnosis

**If NDVI stream is dead (std < 0.01):**
→ Implement **Solution 1** (NDVI as scalar, not image)

**If both streams active but performance poor:**
→ Try longer training OR **Solution 1** anyway

**If actually getting decent R² (≥0.68):**
→ Maybe it's working okay, just not revolutionary

---

## Expected Outcome with Solution 1

**Architecture:**
- RGB stream (ResNet18): 512 features
- NDVI encoder (MLP): 128 features
- Fusion: 640 → 5 predictions

**Expected performance:**
- Similar to baseline (0.51) or slightly better (0.52)
- More honest than fake two-stream
- NDVI properly utilized as scalar feature

**This is essentially what baseline already does with auxiliary tasks!**

---

## Fundamental Problem

**The core issue:** We're trying to create a "two-stream" architecture, but we don't have two actual streams of spatial data!

- **Stream 1:** RGB image (224×224×3) ✓ Has spatial structure
- **Stream 2:** NDVI "image" (224×224×1 of constant value) ✗ NO spatial structure

**Reality:** This is more like "RGB stream + scalar features" than true two-stream.

**Honest architecture would be:**
- RGB CNN for spatial features
- MLP for tabular features (NDVI, height, weather, species, state)
- Fusion of both

**This is what the baseline already does (via auxiliary pretraining)!**

---

## Conclusion

The Two-Stream CNN idea **sounded good in theory** but has a fatal flaw:
- NDVI is a scalar, not a spatial image
- Creating fake spatial image doesn't help
- CNN needs real spatial variation to learn from

**Recommendation:**
1. ✅ Diagnose: Check if NDVI stream is active (run diagnostic code above)
2. ✅ If dead: Implement Solution 1 (NDVI as scalar)
3. ✅ If still poor: Accept that baseline (0.51) is hard to beat with this data
4. ⚠️ Consider: Maybe the training-test distribution gap is the real problem, not architecture

**The real challenge:** Test set has new locations → hard to generalize no matter what architecture!
