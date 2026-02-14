# Heart Disease Classification — Machine Learning Models & Streamlit App

## a) Problem Statement

The objective of this project is to build and compare multiple Machine Learning classification models to predict the presence of heart disease based on patient health features. The project demonstrates a complete ML workflow including preprocessing, model training, evaluation using multiple metrics, comparison of models, and deployment through a Streamlit web application.

Six different classification algorithms are implemented and evaluated to identify the best performing model.

---

## b) Dataset Description

The dataset used is a Heart Disease classification dataset obtained from a public repository.

- Total instances: ~700  
- Total features: 14  
- Problem type: Binary Classification  
- Target column: Last column (Heart disease presence: 0 / 1)

The dataset contains clinical and diagnostic attributes such as age, cholesterol, blood pressure, ECG results, and other medical indicators.

Preprocessing steps performed:

- Converted columns to numeric format
- Handled missing values using median imputation
- Removed invalid values
- Feature scaling applied where required
- Train–test split performed (80–20)

---

## c) Models Used

The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

Each model was evaluated using required metrics:
Accuracy, AUC, Precision, Recall, F1 Score, and MCC.

---

## Model Performance Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|------------|------------|------------|------------|------------|------------|------------|
| Logistic Regression | 0.864286 | 0.945874 | 0.827160 | 0.930556 | 0.875817 | 0.733490 |
| Decision Tree | 0.914286 | 0.915850 | 0.968750 | 0.861111 | 0.911765 | 0.834431 |
| KNN | 0.878571 | 0.930760 | 0.898551 | 0.861111 | 0.879433 | 0.757938 |
| Naive Bayes | 0.864286 | 0.938521 | 0.835443 | 0.916667 | 0.874172 | 0.731263 |
| Random Forest | 0.942857 | 0.993668 | 0.984848 | 0.902778 | 0.942029 | 0.889162 |
| XGBoost | 0.935714 | 0.979984 | 0.970149 | 0.902778 | 0.935252 | 0.873812 |

---

## d) Observations on Model Performance

**Logistic Regression**  
Provided strong baseline performance with high recall. Works well for linear decision boundaries but slightly lower than ensemble models.

**Decision Tree**  
Captured nonlinear relationships effectively and achieved high precision. However, single trees may overfit compared to ensemble methods.

**KNN**  
Performed reasonably well after feature scaling. Sensitive to neighbor selection and distance calculation.

**Naive Bayes**  
Fast and simple model with good recall. Performance is slightly limited due to independence assumption between features.

**Random Forest (Ensemble)**  
Best overall performer on this dataset. Achieved highest accuracy, AUC, F1, and MCC. Ensemble averaging reduced overfitting and improved robustness.

**XGBoost (Ensemble)**  
Second-best performer with very high AUC and precision. Boosting approach improved prediction strength and feature interaction handling.

---

## e) Streamlit App Features

The deployed Streamlit web application includes:

- CSV test dataset upload
- Model selection dropdown
- Automatic preprocessing and scaling
- Display of all required evaluation metrics
- Confusion matrix visualization
- Classification report table
- Prediction preview output

---

## f) Repository Structure

ml_classification_heart_data/
│-- app.py
│-- requirements.txt
│-- README.md
│-- 2025aa05015.ipynb
│-- model/
*.pkl files

---

## g) Deployment

The Streamlit application is deployed using Streamlit Community Cloud directly from this GitHub repository. The app provides an interactive interface to test all trained models on uploaded datasets.

