import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

BASE_DIR = r"F:\ml college programs\cus_churn_prediction"

DATA_PATH = os.path.join(BASE_DIR, "data", "churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.pkl")

df = pd.read_csv(DATA_PATH)

df["Total Charges"] = pd.to_numeric(
    df["Total Charges"], errors="coerce"
)

DROP_COLS = [
    "CustomerID",
    "Churn Label",
    "Churn Score",
    "Churn Reason",
    "Country",
    "City",
    "Zip Code",
    "Lat Long",
    "Count",
    "CLTV"
]

df = df.drop(columns=DROP_COLS)



NUMERIC_FEATURES = [
    "Latitude",
    "Longitude",
    "Tenure Months",
    "Monthly Charges",
    "Total Charges"
]

CATEGORICAL_FEATURES = [
    "State",
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method"
]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ]
)


TARGET = "Churn Value"

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=1000)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

pipeline.fit(X_train, y_train)


os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
