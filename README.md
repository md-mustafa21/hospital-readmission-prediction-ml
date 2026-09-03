# Hospital Readmission Prediction Using Machine Learning

## Thiranex Internship – Project 4

### 📌 Project Overview

This project develops a machine learning system to predict whether a diabetic patient is likely to be readmitted to the hospital within 30 days.

The project uses the **UCI Diabetes 130-US Hospitals for Years 1999–2008** dataset containing over 100,000 hospital encounters.

The main challenge is the significant class imbalance, since early readmissions represent only a small portion of the dataset.

---

## 🎯 Objective

The objective is to:

- Analyze hospital encounter data
- Clean and preprocess healthcare data
- Engineer a binary early-readmission target
- Train classification models
- Handle class imbalance using class weighting and threshold tuning
- Evaluate models using appropriate classification metrics
- Identify important predictive features

---

## 📊 Dataset

**Dataset:** Diabetes 130-US Hospitals for Years 1999–2008

**Source:** UCI Machine Learning Repository

- 101,766 hospital encounters
- 130 U.S. hospitals
- 50 original features
- Target: Readmission within 30 days

The original `readmitted` variable contains:

- `<30` — Readmitted within 30 days
- `>30` — Readmitted after 30 days
- `NO` — Not readmitted

For this project:

- `1` — Early Readmission
- `0` — No Early Readmission

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Checked missing and unknown values
- Replaced `?` values with `Unknown`
- Removed columns with excessive missing values
- Removed patient and encounter identifiers
- Removed constant features
- Converted categorical ID variables into categorical features
- Applied StandardScaler to numerical features
- Applied One-Hot Encoding to categorical features
- Used a stratified 80/20 train-test split

### Target Distribution

- No Early Readmission: **88.84%**
- Early Readmission: **11.16%**

Because of this imbalance, accuracy alone is not sufficient for evaluation.

---

## 🤖 Machine Learning Models

### 1. Logistic Regression

| Metric | Score |
|---|---:|
| Accuracy | 66.1% |
| Precision | 18.0% |
| Recall | 57.3% |
| F1-Score | 27.4% |
| ROC-AUC | 67.0% |

### 2. Random Forest — Default Threshold (0.50)

| Metric | Score |
|---|---:|
| Accuracy | 88.9% |
| Precision | 64.3% |
| Recall | 0.4% |
| F1-Score | 0.8% |
| ROC-AUC | 67.6% |

The high accuracy is misleading because the dataset is highly imbalanced.

### 3. Random Forest — Tuned Threshold (0.15)

| Metric | Score |
|---|---:|
| Accuracy | 77.0% |
| Precision | 21.5% |
| Recall | 40.1% |
| F1-Score | 28.0% |
| ROC-AUC | 67.6% |

The threshold of **0.15** produced the highest F1-score among the tested Random Forest thresholds.

---

## 📈 Evaluation

The project uses:

- Confusion Matrix
- ROC Curve
- ROC-AUC
- Precision
- Recall
- F1-Score
- Threshold Tuning
- Feature Importance
- Sample Prediction

---

## 🔍 Important Predictive Features

The strongest predictive signals identified by the Random Forest included:

1. Number of lab procedures
2. Number of medications
3. Number of inpatient visits
4. Time in hospital
5. Number of diagnoses
6. Number of procedures
7. Number of outpatient visits
8. Number of emergency visits

These features suggest that hospital utilization, treatment intensity, and patient complexity contain useful predictive information.

**Note:** Feature importance represents predictive contribution and does not imply causation.

---

## 💡 Key Insights

- Early readmission represents only **11.16%** of the dataset.
- Accuracy alone can therefore be misleading.
- Logistic Regression achieved higher recall than the default Random Forest.
- Threshold tuning substantially improved Random Forest recall.
- The tuned Random Forest achieved the highest F1-score among the evaluated Random Forest thresholds.
- Hospital utilization and treatment-related variables were among the strongest predictive signals.

---

## ⚠️ Limitations

- The dataset contains historical hospital encounter data.
- The ROC-AUC indicates moderate predictive discrimination rather than highly accurate prediction.
- Some features may represent information available later in the hospital encounter, limiting strict real-time admission prediction.
- The project is intended for educational and research purposes.
- It should not be used as a clinical diagnostic or decision-support system.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Google Colab
- Jupyter Notebook

---

## 📁 Project Files

```text
hospital-readmission-prediction-ml/
│
├── Hospital_Readmission_Prediction_ML.ipynb
├── hospital_readmission_random_forest.pkl
├── readmission_preprocessor.pkl
├── cleaned_diabetes_readmission_dataset.csv
└── README.md
