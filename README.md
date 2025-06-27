# Logistic Regression Classifier: Breast Cancer Detection

This project implements a **binary classification** system using **Logistic Regression** on the **Breast Cancer Wisconsin Diagnostic Dataset**. The goal is to predict whether a tumor is **malignant** or **benign** based on 30 medical features.

---

# Dataset Overview

- **Target**: `diagnosis` (M = Malignant, B = Benign)
- **Features**: 30 numerical features like radius, texture, perimeter, etc.
- **Rows**: 569 samples

---

#Pipeline Steps

1. **Data Preprocessing**
   - Dropped irrelevant columns: `id`, `Unnamed: 32`
   - Encoded target (`M` → 1, `B` → 0)
   - Scaled features with `StandardScaler`

2. **Model Training**
   - Logistic Regression (`sklearn.linear_model.LogisticRegression`)
   - Train-test split (80-20)

3. **Model Evaluation**
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1-score)
   - ROC-AUC Score & Curve
   - Precision-Recall vs Threshold plot

---

# Results

- **Accuracy**: 96%
- **ROC-AUC Score**: 0.996
- **Confusion Matrix**:
