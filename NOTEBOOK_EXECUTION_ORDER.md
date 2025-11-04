# Notebook Execution Order

## 06_teacher_student_comparison.ipynb

### Correct Cell Execution Order:

1. **Setup Notes** (markdown) - Caching and debug mode info
2. **Part 1: Setup & Data Preparation** (markdown header)
3. **Imports** - All libraries and device setup
4. **Debug Mode Configuration** - Set `DEBUG_MODE = True/False`
5. **Load Data** - Read train_enriched.csv
6. **Define Features** - Target columns and feature lists
7. **Train/Val Split** - Split data 80/20
8. **Scalers** - StandardScaler and LabelEncoder
9. **Dataset Classes** (markdown header)
10. **PastureDataset** - Dataset class definition
11. **BaselineModel** - CNN architecture
12. **CompetitionLoss** - Weighted loss function
13. **Training Utilities** - `train_model()` and `evaluate_model()` functions ⚠️ MUST RUN BEFORE TRAINING
14. **Part 2: Baseline** (markdown header)
15. **Train Baseline** - Trains baseline model
16. **Evaluate Baseline** - Tests baseline performance

17. **Part 3: Teacher** (markdown header)
18. **TeacherModel** - Multimodal architecture
19. **Teacher Training Functions** - `train_teacher()` and `evaluate_teacher()`
20. **Train Teacher** - Trains teacher model
21. **Evaluate Teacher** - Tests teacher performance

22. **Part 4: Student** (markdown header)
23. **StudentModel** - Alias for BaselineModel
24. **DistillationLoss** - Triple loss for distillation
25. **Student Training Function** - `train_student_distillation()`
26. **Train Student** - Distills from teacher
27. **Evaluate Student** - Tests student performance

28. **Part 5: Auxiliary** (markdown header)
29. **AuxiliaryModel** - Multi-head architecture
30. **AuxiliaryLoss** - Multi-task loss
31. **Auxiliary Training Function** - `train_auxiliary()`
32. **Train Auxiliary** - Trains with auxiliary tasks
33. **Evaluate Auxiliary** - Tests auxiliary performance

34. **Part 6: Comparison** (markdown header)
35. **Comparison Table** - Results dataframe
36. **Visualization** - Bar charts and R² comparison
37. **Summary** (markdown) - Conclusions and next steps

## Key Points:

- **Cell 13 is critical**: Contains `train_model()` and `evaluate_model()` - must run before any training
- **Debug mode**: Set in Cell 4 to control epoch counts
- **Model checkpoints**: All models save `.pth` files automatically
- **Outputs cache**: VSCode Jupyter automatically caches outputs if you save the notebook

## Quick Test (Debug Mode):

1. Set `DEBUG_MODE = True` in Cell 4
2. Run All Cells
3. Each model trains for 1 epoch (~2-3 minutes each)
4. Total runtime: ~10-15 minutes

## Full Training:

1. Set `DEBUG_MODE = False` in Cell 4
2. Run All Cells
3. Baseline: 10 epochs (~20-30 min)
4. Teacher: 15 epochs (~30-45 min)
5. Student: 15 epochs (~30-45 min)
6. Auxiliary: 15 epochs (~30-45 min)
7. Total runtime: ~2-3 hours
