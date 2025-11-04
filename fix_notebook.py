#!/usr/bin/env python3
"""
Script to create a properly ordered notebook from the scrambled one.
"""
import json

# Load the scrambled notebook
with open('06_teacher_student_comparison.ipynb', 'r') as f:
    old_nb = json.load(f)

# Create new notebook structure
new_nb = {
    'cells': [],
    'metadata': old_nb.get('metadata', {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        }
    }),
    'nbformat': 4,
    'nbformat_minor': 4
}

# Define the correct order by extracting cells we need (avoiding duplicates)
# I'll manually specify which cell index from the old notebook to use

cell_order = [
    0,   # Title markdown
    1,   # Setup notes markdown
    2,   # Part 1 header
    3,   # Imports
    4,   # Debug config
    5,   # Load data
    6,   # Define features
    7,   # Train/val split
    8,   # Scalers
    9,   # Dataset classes header
    10,  # PastureDataset class
    11,  # BaselineModel
    12,  # CompetitionLoss
    13,  # Part 2 header (was added correctly)
    14,  # Training utilities
    15,  # Train Baseline (FIRST occurrence, skip 16 which is duplicate)
    17,  # Evaluate Baseline
    18,  # Part 3 header
    19,  # TeacherModel
    20,  # Teacher training functions
    37,  # Train Teacher (cell 37, skip 38 duplicate)
    # Need to find Evaluate Teacher - might be missing
    36,  # Part 4 header
    35,  # StudentModel
    34,  # DistillationLoss
    33,  # Student training function
    31,  # Train Student (cell 31, skip 32 duplicate)
    # Need to find Evaluate Student
    30,  # Part 5 header
    29,  # AuxiliaryModel
    28,  # AuxiliaryLoss
    27,  # Auxiliary training function
    25,  # Train Auxiliary (cell 25, skip 26 duplicate)
    # Need to find Evaluate Auxiliary
    24,  # Part 6 header
    23,  # Comparison table
    22,  # Visualization
    21,  # Summary markdown
]

print(f"Old notebook has {len(old_nb['cells'])} cells")
print(f"Will create new notebook with {len(cell_order)} cells")

# Extract cells in correct order
for idx in cell_order:
    if idx < len(old_nb['cells']):
        new_nb['cells'].append(old_nb['cells'][idx])
        source = ''.join(old_nb['cells'][idx].get('source', []))
        first_line = source.split('\n')[0][:60]
        print(f"  {len(new_nb['cells'])-1}: From old[{idx}] - {first_line}")
    else:
        print(f"  WARNING: Cell {idx} doesn't exist in old notebook!")

# Save the new notebook
with open('06_teacher_student_comparison_fixed.ipynb', 'w') as f:
    json.dump(new_nb, f, indent=2)

print(f"\n✓ Created new notebook: 06_teacher_student_comparison_fixed.ipynb")
print(f"  New notebook has {len(new_nb['cells'])} cells")
print(f"\nPlease review and then:")
print(f"  mv 06_teacher_student_comparison.ipynb 06_teacher_student_comparison_broken.ipynb")
print(f"  mv 06_teacher_student_comparison_fixed.ipynb 06_teacher_student_comparison.ipynb")
