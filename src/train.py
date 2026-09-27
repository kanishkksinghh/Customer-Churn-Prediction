import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. PROJECT PATHS
# ============================================================

DATA_PATH = "data/customer_churn.csv"
MODEL_DIR = "models"


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(r"D:\Machine Learning\Customer Churn Prediction\datacustomer_churn.csv")

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# 3. SELECT FEATURES AND TARGET
# ============================================================

features = [
    "Age",
    "Total_Purchase",
    "Account_Manager",
    "Years",
    "Num_Sites"
]

target = "Churn"

X = df[features]
y = df[target]

print("\nFeatures:")
print(features)

print("\nTarget:")
print(target)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain/Test Split:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# ============================================================
# 5. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

# Learn scaling parameters only from training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same transformation to test data
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")


# ============================================================
# 6. LOGISTIC REGRESSION
# ============================================================

logistic_model = LogisticRegression(
    random_state=42
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

print("\nLogistic Regression trained successfully.")


# ============================================================
# 7. RANDOM FOREST
# ============================================================

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Random Forest does not require scaled features
random_forest_model.fit(
    X_train,
    y_train
)

print("Random Forest trained successfully.")


# ============================================================
# 8. CREATE MODELS DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 9. SAVE MODELS
# ============================================================

joblib.dump(
    logistic_model,
    os.path.join(
        MODEL_DIR,
        "logistic_regression.pkl"
    )
)

joblib.dump(
    random_forest_model,
    os.path.join(
        MODEL_DIR,
        "random_forest.pkl"
    )
)

joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )
)


# ============================================================
# 10. FINISHED
# ============================================================

print("\n===================================")
print("Training completed successfully!")
print("===================================")

print("\nSaved files:")

print("models/logistic_regression.pkl")
print("models/random_forest.pkl")
print("models/scaler.pkl")