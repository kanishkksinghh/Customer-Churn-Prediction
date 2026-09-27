# Customer-Churn-Prediction

````markdown
# Customer Churn Prediction

A machine learning project that analyzes customer behavior and predicts whether a customer is likely to churn.

The project covers the complete machine learning workflow — from exploratory data analysis and preprocessing to model training, evaluation, feature importance, and customer-level churn prediction.

---

## Project Overview

Customer churn occurs when a customer stops using a company's products or services.

The objective of this project is to analyze customer behavior and build machine learning models that can identify customers who are more likely to churn.

### Project Pipeline

```text
Raw Customer Data
        │
        ▼
Data Understanding
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Selection
        │
        ▼
Train / Test Split
        │
        ▼
Feature Scaling
        │
        ├─────────────────────┐
        ▼                     ▼
Logistic Regression     Random Forest
        │                     │
        └──────────┬──────────┘
                   ▼
            Model Evaluation
                   │
                   ▼
          Feature Importance
                   │
                   ▼
        Customer-Level Prediction
````

---

# Dataset

The dataset contains **900 customer records** and the following 10 columns:

| Feature           | Description                                  |
| ----------------- | -------------------------------------------- |
| `Names`           | Customer name                                |
| `Age`             | Customer age                                 |
| `Total_Purchase`  | Total customer purchase value                |
| `Account_Manager` | Whether the customer has an account manager  |
| `Years`           | Number of years as a customer                |
| `Num_Sites`       | Number of sites associated with the customer |
| `Onboard_date`    | Customer onboarding date                     |
| `Location`        | Customer location                            |
| `Company`         | Customer company                             |
| `Churn`           | Target variable                              |

### Target Variable

```text
Churn = 0 → Customer stayed
Churn = 1 → Customer churned
```

The dataset contains approximately:

```text
750 retained customers
150 churned customers
16.67% churn rate
```

---

# Exploratory Data Analysis

The EDA phase was used to understand customer behavior and identify patterns related to churn.

The analysis includes:

* Churn distribution
* Customer age distribution
* Purchase behavior
* Customer tenure
* Number of sites
* Account manager relationship
* Location-based churn
* Feature correlations
* Churn vs numerical variables

Example visualizations are stored in:

```text
photos/
```

---

# Machine Learning

Two classification models are implemented.

## 1. Logistic Regression

Logistic Regression is used as the baseline classification model.

Pipeline:

```text
Customer Features
       ↓
StandardScaler
       ↓
Logistic Regression
       ↓
Churn Prediction
```

The model also provides a probability of churn using:

```python
model.predict_proba()
```

---

## 2. Random Forest

Random Forest is used as a second classification model.

It is useful for:

* Non-linear relationships
* Classification
* Feature importance analysis
* Comparing against the baseline model

Random Forest does not require feature scaling for this project.

---

# Features Used

The baseline models use the following features:

```text
Age
Total_Purchase
Account_Manager
Years
Num_Sites
```

Other columns such as `Names`, `Company`, `Location`, and `Onboard_date` are not directly included in the initial baseline model.

Feature engineering was also explored using the onboarding date and purchase-per-year information.

---

# Data Preprocessing

The preprocessing stage includes:

* Checking missing values
* Checking duplicate records
* Converting date values
* Selecting relevant features
* Separating features and target
* Train/test splitting
* Feature scaling

### Train/Test Split

The dataset is divided into:

```text
80% → Training
20% → Testing
```

The split uses stratification:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

`stratify=y` helps maintain a similar churn distribution in both training and testing data.

---

# Feature Scaling

Logistic Regression uses `StandardScaler`.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data to avoid data leakage.

---

# Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

### Why multiple metrics?

The dataset contains more customers who stayed than customers who churned.

Therefore, accuracy alone may not give a complete picture of model performance.

Precision and recall are especially useful for understanding how well the model identifies churned customers.

---

# Model Comparison

The final model comparison will contain the actual results obtained from the trained models.

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |        — |         — |      — |        — |       — |
| Random Forest       |        — |         — |      — |        — |       — |

> Metrics will be updated after the final evaluation pipeline is completed.

---

# Feature Importance

Random Forest provides feature importance values that can be used to understand which customer attributes contribute most to the model's predictions.

Example output:

```text
Feature              Importance
--------------------------------
Num_Sites            ...
Years                ...
Total_Purchase       ...
Age                  ...
Account_Manager      ...
```

The actual values are generated from the trained model.

---

# Customer-Level Prediction

The project also supports prediction for an individual customer.

Example:

```python
prediction, probability = predict_churn(
    age=45,
    total_purchase=12000,
    account_manager=1,
    years=6,
    num_sites=12
)
```

The model returns:

```text
Churn Prediction
Churn Probability
```

This allows the project to move beyond dataset-level analysis toward an individual customer risk prediction workflow.

---

# Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── customer_churn.csv
│
├── notebooks/
│   └── customer_churn.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
│
├── photos/
│   ├── churn_distribution.png
│   ├── correlation_matrix.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── model_comparison.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Model Persistence

* Joblib

### Development

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

# Installation

Clone the repository:

```bash
git clone <your-repository-url>
```

Move into the project:

```bash
cd customer-churn-prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run the Project

Make sure the dataset is located at:

```text
data/customer_churn.csv
```

Train the models:

```bash
python src/train.py
```

This generates:

```text
models/
├── logistic_regression.pkl
├── random_forest.pkl
└── scaler.pkl
```

Run the notebook to explore the complete EDA and machine learning workflow.

---

# Key Concepts Demonstrated

This project demonstrates practical understanding of:

* Exploratory Data Analysis
* Data preprocessing
* Feature selection
* Feature engineering
* Numerical and categorical features
* Train/test splitting
* Stratified sampling
* Feature scaling
* Logistic Regression
* Random Forest
* Binary classification
* Model evaluation
* Confusion matrices
* Precision and recall
* F1-score
* ROC-AUC
* Feature importance
* Probability-based predictions
* Data leakage prevention
* Model persistence with Joblib

---

# Future Improvements

Possible improvements include:

* One-hot encoding for categorical features
* Cross-validation
* Hyperparameter tuning
* XGBoost comparison
* Threshold optimization
* SHAP explainability
* Customer risk segmentation
* Streamlit prediction dashboard
* Automated model evaluation
* Model monitoring

---

# Project Status

```text
[x] Dataset exploration
[x] Data cleaning
[x] Exploratory Data Analysis
[x] Feature selection
[x] Feature engineering
[x] Train/test split
[x] Logistic Regression
[x] Random Forest
[x] Model training
[x] Model persistence
[ ] Final evaluation report
[ ] Feature importance visualization
[ ] Final model comparison
[ ] Customer prediction interface
[ ] Streamlit dashboard
```

---

# Learning Outcome

This project demonstrates the complete workflow of developing a binary classification machine learning solution:

```text
Understand Data
      ↓
Explore Data
      ↓
Prepare Data
      ↓
Train Models
      ↓
Evaluate Models
      ↓
Interpret Results
      ↓
Make Predictions
```

The focus of the project is not only on obtaining predictions, but also on understanding the reasoning behind each stage of the machine learning pipeline.

---

# Author

**Kanishk Singh**

Computer Science Student
Machine Learning | Data Science | Software Development

---

## License

This project is intended for educational and portfolio purposes.

```

### One important thing before pushing this to GitHub

Don't leave the model comparison table with made-up numbers. Once we finish `evaluate.py`, we'll put your **actual Logistic Regression and Random Forest results** there.

Also, I would **not commit `*.pkl` files blindly** if they're large. Your `.gitignore` can keep them out of GitHub while `train.py` regenerates them.
```
