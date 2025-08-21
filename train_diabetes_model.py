import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix, roc_auc_score


df = pd.read_csv('data/diabetes.csv')

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET = "Outcome"

X = df[FEATURES].copy()
y = df[TARGET].astype(int)

zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
X[zero_as_missing] = X[zero_as_missing].replace(0, np.nan)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
])

pipeline.fit(x_train,y_train)
y_pred = pipeline.predict(x_test)
y_proba = pipeline.predict_proba(x_test)[:,1]

print("\n=== Test Metrics ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("ROC AUC:", round(roc_auc_score(y_test, y_proba), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


dump({
    "pipeline": pipeline,
    "features": FEATURES
}, "diabetes_pipeline.joblib")