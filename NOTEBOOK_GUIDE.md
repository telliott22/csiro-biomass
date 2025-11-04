# Complete Notebook Guide: 06_teacher_student_comparison.ipynb

## Current Status
The notebook has been created with:
✅ Setup and data loading (cells 1-7)
✅ BaselineModel definition
✅ CompetitionLoss definition
✅ Training utilities
✅ Baseline training cell
✅ Baseline evaluation cell
✅ TeacherModel definition

## Remaining Cells to Add

### Teacher Model Training (Add these cells):

```python
# Train Teacher Model with multimodal data
def train_teacher(model, train_loader, val_loader, criterion, num_epochs=15):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = batch['image'].to(device)
            weather = batch['weather'].to(device)
            ndvi_height = batch['ndvi_height'].to(device)
            state = batch['state'].to(device)
            species = batch['species'].to(device)
            targets = batch['targets'].to(device)

            optimizer.zero_grad()
            outputs = model(images, weather, ndvi_height, state, species)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                weather = batch['weather'].to(device)
                ndvi_height = batch['ndvi_height'].to(device)
                state = batch['state'].to(device)
                species = batch['species'].to(device)
                targets = batch['targets'].to(device)

                outputs = model(images, weather, ndvi_height, state, species)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"  Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'teacher_best.pth')
            print("  ✓ Saved")

# Create teacher
train_loader_teacher = DataLoader(train_dataset_teacher, batch_size=16, shuffle=True, num_workers=0)
val_loader_teacher = DataLoader(val_dataset_teacher, batch_size=16, shuffle=False, num_workers=0)

teacher_model = TeacherModel(num_outputs=5, num_states=4, num_species=len(le_species.classes_)).to(device)
train_teacher(teacher_model, train_loader_teacher, val_loader_teacher, competition_loss, num_epochs=15)
```

### Student Model + Distillation:

```python
# Student Model (same as baseline but will learn from teacher)
student_model = BaselineModel(num_outputs=5).to(device)

# Distillation Loss
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.5, gamma=0.2, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.T = temperature
        self.competition_loss = CompetitionLoss()

    def forward(self, student_out, teacher_out, targets, student_feat, teacher_feat):
        # Hard loss: student vs ground truth
        hard_loss = self.competition_loss(student_out, targets)

        # Soft loss: student vs teacher (temperature scaled)
        soft_loss = F.mse_loss(student_out / self.T, teacher_out / self.T) * (self.T ** 2)

        # Feature loss: match CNN features
        feature_loss = F.mse_loss(student_feat, teacher_feat.detach())

        total_loss = self.alpha * hard_loss + self.beta * soft_loss + self.gamma * feature_loss
        return total_loss

# Train student via distillation
def train_student_distillation(student, teacher, train_loader, val_loader, num_epochs=15):
    teacher.eval()  # Freeze teacher
    distill_loss = DistillationLoss().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=1e-4)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        student.train()
        train_loss = 0

        for batch_simple, batch_teacher in zip(train_loader_simple, train_loader_teacher):
            images = batch_simple['image'].to(device)
            targets = batch_simple['targets'].to(device)

            # Get teacher predictions (no gradient)
            with torch.no_grad():
                teacher_out, teacher_feat = teacher(
                    batch_teacher['image'].to(device),
                    batch_teacher['weather'].to(device),
                    batch_teacher['ndvi_height'].to(device),
                    batch_teacher['state'].to(device),
                    batch_teacher['species'].to(device),
                    return_features=True
                )

            # Student predictions
            optimizer.zero_grad()
            student_feat = student.get_features(images)
            student_out = student(images)

            loss = distill_loss(student_out, teacher_out, targets, student_feat, teacher_feat)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader_simple.dataset)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")

        # Save best
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            torch.save(student.state_dict(), 'student_best.pth')

# Train
train_student_distillation(student_model, teacher_model, train_loader_simple, train_loader_teacher, num_epochs=15)
```

### Auxiliary Multi-Task Model:

```python
# Auxiliary Model
class AuxiliaryModel(nn.Module):
    def __init__(self, num_outputs=5):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Shared features
        self.shared = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4)
        )

        # Main head: biomass
        self.biomass_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs)
        )

        # Auxiliary heads
        self.ndvi_head = nn.Linear(512, 1)
        self.height_head = nn.Linear(512, 1)
        self.temp_head = nn.Linear(512, 1)
        self.rain_head = nn.Linear(512, 1)

    def forward(self, x, return_auxiliary=False):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        x = self.shared(x)

        biomass = self.biomass_head(x)

        if return_auxiliary:
            return {
                'biomass': biomass,
                'ndvi': self.ndvi_head(x),
                'height': self.height_head(x),
                'temp': self.temp_head(x),
                'rainfall': self.rain_head(x)
            }
        return biomass

# Auxiliary Loss
class AuxiliaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.competition_loss = CompetitionLoss()

    def forward(self, outputs, targets, aux_targets):
        biomass_loss = self.competition_loss(outputs['biomass'], targets)
        ndvi_loss = F.mse_loss(outputs['ndvi'].squeeze(), aux_targets[:, 0])
        height_loss = F.mse_loss(outputs['height'].squeeze(), aux_targets[:, 1])
        temp_loss = F.mse_loss(outputs['temp'].squeeze(), aux_targets[:, 2])
        rain_loss = F.mse_loss(outputs['rainfall'].squeeze(), aux_targets[:, 3])

        total = biomass_loss + 0.2*ndvi_loss + 0.2*height_loss + 0.1*temp_loss + 0.1*rain_loss
        return total

# Train auxiliary model
train_loader_auxiliary = DataLoader(train_dataset_auxiliary, batch_size=16, shuffle=True, num_workers=0)
val_loader_auxiliary = DataLoader(val_dataset_auxiliary, batch_size=16, shuffle=False, num_workers=0)

auxiliary_model = AuxiliaryModel().to(device)
aux_loss = AuxiliaryLoss()

# Training loop similar to baseline but with auxiliary outputs
# ... (add full training loop)
```

### Final Comparison:

```python
# Compare all models
results = {
    'Baseline': baseline_score,
    'Teacher': teacher_score,  # (multimodal, not usable at test)
    'Student': student_score,
    'Auxiliary': auxiliary_score
}

# Plot comparison
plt.figure(figsize=(10, 6))
models = list(results.keys())
scores = list(results.values())
plt.bar(models, scores)
plt.ylabel('Competition Score (Weighted R²)')
plt.title('Model Comparison')
plt.ylim([0, 1])
plt.grid(axis='y', alpha=0.3)
plt.show()

print("\\nRanking:")
for i, (model, score) in enumerate(sorted(results.items(), key=lambda x: x[1], reverse=True), 1):
    print(f"{i}. {model}: {score:.4f}")
```

## Quick Start

To complete the notebook, add the cells above in this order:
1. Teacher training
2. Student + distillation
3. Auxiliary model
4. Final comparison

Each section is ~10-20 minutes to train, ~1 hour total.

## Expected Results

| Model | Competition Score | Notes |
|-------|------------------|-------|
| Teacher | 0.70-0.75 | Best (but needs tabular data) |
| Student | 0.65-0.72 | 85-95% of teacher performance |
| Auxiliary | 0.60-0.68 | Simpler approach |
| Baseline | 0.50-0.60 | No knowledge transfer |

Winner should be **Student (distilled)** or **Auxiliary**, both are image-only and deployable!
