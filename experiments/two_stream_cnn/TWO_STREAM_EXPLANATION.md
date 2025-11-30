# Two-Stream CNN for Biomass Prediction

## What is a Two-Stream CNN?

A **Two-Stream Convolutional Neural Network** is an architecture that processes the same input through two parallel CNN pathways, each learning complementary features. The outputs are then fused for final prediction.

Originally developed for video action recognition, where:
- **Spatial Stream**: Processes RGB frames (appearance)
- **Temporal Stream**: Processes optical flow (motion)

## Why Two-Stream for Biomass Prediction?

### Problem with Single-Stream Approaches

**Current results:**
- Baseline (single ResNet18): **0.51**
- K-Fold ensemble (all features): **0.50**
- K-Fold ensemble (universal features): **0.40**

**Single-stream limitation:**
- CNN sees only RGB pixels
- Must learn everything from visual appearance
- NDVI treated as tabular feature (auxiliary task)
- No explicit vegetation health modeling

### Two-Stream Solution

**Stream 1: RGB Appearance Stream**
- Input: RGB image (224×224×3)
- Backbone: ResNet18
- Output: 512-dim appearance features
- **Learns**: Color, texture, structure, plant density

**Stream 2: Vegetation Health Stream**
- Input: NDVI image (224×224×1)
- Backbone: ResNet18 (modified for 1-channel input)
- Output: 512-dim vegetation features
- **Learns**: Vegetation density patterns, health indicators

**Fusion Layer**
- Concatenates both feature streams
- Combined: 1024-dim features
- MLP layers → 5 biomass predictions

## Architecture Diagram

```
                        INPUT
                          |
            +-------------+-------------+
            |                           |
     RGB Image (3ch)              NDVI Image (1ch)
            |                           |
    +-------v-------+           +-------v-------+
    | RGB Stream    |           | NDVI Stream   |
    | (ResNet18)    |           | (ResNet18)    |
    +-------+-------+           +-------+-------+
            |                           |
       512 features                512 features
            |                           |
            +-------------+-------------+
                          |
                   Concatenate
                          |
                   1024 features
                          |
                  +-------v-------+
                  | Fusion MLP    |
                  | (1024→512→256)|
                  +-------+-------+
                          |
                    5 predictions
            (Dry_Green, Dry_Dead, Dry_Clover, GDM, Dry_Total)
```

## Implementation Details

### NDVI Image Creation

**Challenge:** NDVI is a single scalar value per sample (e.g., 0.75), not a spatial image.

**Solution:** Create pseudo-image where every pixel = NDVI value

```python
ndvi_value = 0.75  # From tabular data
ndvi_image = torch.full((224, 224), ndvi_value)  # 224×224 array of 0.75
ndvi_image = ndvi_image.unsqueeze(0)  # Add channel: (1, 224, 224)
```

**Rationale:**
- Lets NDVI stream learn "if NDVI=0.75, expect X biomass"
- Maintains spatial dimensions for CNN processing
- Can extend later with actual NDVI rasters if available

### Model Architecture

```python
class TwoStreamModel(nn.Module):
    def __init__(self, num_outputs=5, hidden_dim=512, dropout=0.3):
        super().__init__()

        # Stream 1: RGB (3-channel input)
        resnet_rgb = models.resnet18(weights=None)
        self.rgb_stream = nn.Sequential(*list(resnet_rgb.children())[:-1])

        # Stream 2: NDVI (1-channel input)
        resnet_ndvi = models.resnet18(weights=None)
        # Modify first conv layer for 1-channel input
        resnet_ndvi.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                      padding=3, bias=False)
        self.ndvi_stream = nn.Sequential(*list(resnet_ndvi.children())[:-1])

        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(512 + 512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_outputs)
        )

    def forward(self, rgb_img, ndvi_img):
        # Extract features from both streams
        rgb_features = self.rgb_stream(rgb_img).flatten(1)    # [batch, 512]
        ndvi_features = self.ndvi_stream(ndvi_img).flatten(1)  # [batch, 512]

        # Concatenate and fuse
        combined = torch.cat([rgb_features, ndvi_features], dim=1)  # [batch, 1024]
        output = self.fusion(combined)  # [batch, 5]

        return output
```

### Dataset Modifications

```python
class TwoStreamDataset(Dataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. RGB image
        rgb_img = Image.open(row['full_image_path']).convert('RGB')
        rgb_img = self.rgb_transform(rgb_img)  # → [3, 224, 224]

        # 2. NDVI image (create from scalar value)
        ndvi_value = row['Pre_GSHH_NDVI']
        ndvi_img = torch.full((224, 224), ndvi_value, dtype=torch.float32)
        ndvi_img = ndvi_img.unsqueeze(0)  # → [1, 224, 224]
        # Normalize (optional)
        ndvi_img = (ndvi_img - 0.5) / 0.2  # Assuming NDVI range [0, 1]

        # 3. Targets
        targets = torch.tensor(row[TARGET_COLS].values, dtype=torch.float32)
        targets_normalized = (targets - self.target_means) / self.target_stds

        return {
            'rgb_image': rgb_img,
            'ndvi_image': ndvi_img,
            'targets': targets_normalized,
            'targets_original': targets
        }
```

## Training Strategy

### Approach: End-to-End Training

**Simple and effective:**
1. Initialize both streams randomly (no ImageNet pretraining)
2. Train entire network jointly
3. Single loss function (MSE on biomass predictions)
4. 30 epochs, learning rate 3e-4

**Configuration:**
```python
CONFIG = {
    'epochs': 30,
    'batch_size': 16,
    'learning_rate': 3e-4,
    'weight_decay': 1e-4,
    'hidden_dim': 512,
    'dropout': 0.3,
    'rgb_backbone': 'resnet18',
    'ndvi_backbone': 'resnet18',
}
```

### Alternative: Two-Phase Training (if end-to-end fails)

**Phase 1: Pretrain streams separately**
- Train RGB stream on biomass (10 epochs)
- Train NDVI stream on biomass (10 epochs)

**Phase 2: Joint fine-tuning**
- Freeze backbones, train fusion only (10 epochs)
- Unfreeze all, fine-tune end-to-end (10 epochs)

## Expected Performance

### Optimistic Scenario
- **Kaggle Score: 0.53-0.55**
- Two complementary views capture more information
- NDVI stream provides vegetation-specific patterns
- Fusion combines best of both

### Realistic Scenario
- **Kaggle Score: 0.52-0.53**
- Modest improvement over baseline (0.51)
- Architecture benefits offset by same training data

### Pessimistic Scenario
- **Kaggle Score: 0.51 (matches baseline)**
- NDVI-as-image doesn't add value
- Increased complexity without benefit

## Advantages Over Previous Attempts

| Approach | Score | Issue |
|----------|-------|-------|
| Baseline (single stream) | 0.51 | Single perspective |
| K-Fold (all features) | 0.50 | Overfitting |
| K-Fold (universal) | 0.40 | Missing signal |
| **Two-Stream** | **?** | **Dual perspective** |

**Why Two-Stream might succeed:**
1. ✅ **Explicit vegetation modeling** - NDVI has dedicated pathway
2. ✅ **Feature specialization** - Each stream focuses on one aspect
3. ✅ **No K-Fold complexity** - Single model, simpler training
4. ✅ **Novel approach** - Haven't tried multi-stream yet
5. ✅ **Low risk** - Fast to train (~2 hours)

## Future Extensions

### Three-Stream Architecture

If two-stream works, could add third stream:

**Stream 1:** RGB appearance (512 features)
**Stream 2:** NDVI vegetation (512 features)
**Stream 3:** Auxiliary features (species, state, weather) → MLP → 512 features

**Fusion:** 1536 combined features → biomass predictions

### Attention-Based Fusion

Instead of simple concatenation, learn which stream to weight:

```python
attention_weights = softmax(W @ [rgb_feat, ndvi_feat])
combined = attention_weights[0] * rgb_feat + attention_weights[1] * ndvi_feat
```

### Multi-Scale Fusion

Combine features at multiple ResNet layers:
- Early fusion: After conv2 blocks
- Mid fusion: After conv4 blocks
- Late fusion: After final layer

## Success Criteria

**Minimum Success: 0.51**
- Matches baseline
- Validates two-stream architecture
- Foundation for future improvements

**Target Success: 0.52-0.53**
- Clear improvement over baseline
- Justifies multi-stream approach
- Worth further exploration

**Stretch Success: 0.54+**
- Significant improvement
- Strong evidence for architecture
- Pursue three-stream extension

## Comparison to Other Architectures

| Architecture | Parameters | Score | Training Time |
|--------------|------------|-------|---------------|
| ResNet18 (baseline) | 11M | 0.51 | 1h |
| ResNet50 | 25M | 0.51 | 2h |
| K-Fold (5× ResNet18) | 55M | 0.50 | 4h |
| **Two-Stream ResNet18** | **22M** | **?** | **2h** |

**Two-Stream characteristics:**
- 2× parameters (two backbones)
- Same training time as ResNet50
- Different approach (architecture vs capacity)

## Implementation Files

1. **28_two_stream_test.ipynb** - Main training notebook
2. **29_two_stream_submission.ipynb** - Kaggle submission (if successful)
3. **TWO_STREAM_EXPLANATION.md** - This file

## References

- Simonyan & Zisserman (2014): "Two-Stream Convolutional Networks for Action Recognition in Videos"
- Multi-stream CNNs in agricultural remote sensing
- Fusion strategies for multi-modal learning

---

**Status:** Ready for implementation
**Expected completion:** Tonight
**Next step:** Create training notebook
