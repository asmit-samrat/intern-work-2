# Predictive Modeling Using Machine Learning

## Student Performance Prediction

### Objective
Build supervised machine-learning models to predict whether a student will pass using study, attendance, assignment and previous-performance features.

### Dataset
The project includes **600 records** with:
- `study_hours`
- `attendance_percent`
- `assignment_score`
- `previous_score`
- `sleep_hours`
- `internet_access`
- `passed` — target (0 = Fail, 1 = Pass)

The dataset is a generated educational dataset so the complete project can run immediately. For a real-world submission, replace it with a cited real dataset if your instructor requires one.

### Algorithms
- Logistic Regression
- Decision Tree
- Random Forest

### Workflow
1. Load and inspect data
2. Check missing values
3. Split data into train/test sets
4. Scale features for Logistic Regression
5. Train three supervised-learning models
6. Compare Accuracy, Precision, Recall, F1 and ROC-AUC
7. Generate a confusion matrix
8. Plot ROC curves
9. Select the best model

### Final Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.733 | 0.767 | 0.863 | 0.812 | 0.776 |\n| Logistic Regression | 0.742 | 0.753 | 0.912 | 0.825 | 0.744 |\n| Decision Tree | 0.675 | 0.753 | 0.762 | 0.758 | 0.716 |\n
### Best Model
**Random Forest** achieved the highest ROC-AUC of **0.776**.

### How to Run
```bash
pip install -r requirements.txt
python predictive_modeling.py
```

Or open `predictive_modeling.ipynb` in Jupyter Notebook.

### Files
- `student_performance.csv` — dataset
- `predictive_modeling.py` — complete Python program
- `predictive_modeling.ipynb` — notebook
- `model_results.csv` — evaluation table
- `confusion_matrix.png` — confusion matrix
- `roc_curves.png` — ROC curves
- `model_comparison.png` — model comparison
- `report.md` — submission report
