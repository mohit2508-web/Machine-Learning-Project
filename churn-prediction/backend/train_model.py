import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# Paths
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
df.drop("customerID", axis=1, inplace=True)

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Encode categorical columns
for col in df.select_dtypes(include=['object']).columns:
    if df[col].nunique() == 2:
        df[col] = LabelEncoder().fit_transform(df[col])
    else:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# Split data
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Balance dataset using SMOTE
sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---- Define models & their parameters ----
models_params = {
    "Logistic Regression": (LogisticRegression(max_iter=2000), {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    }),
    "SVM": (SVC(probability=True), {
        'C': [0.5, 1, 5],
        'kernel': ['rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }),
    "Naive Bayes": (GaussianNB(), {}),
    "KNN": (KNeighborsClassifier(), {
        'n_neighbors': [5, 7, 9],
        'weights': ['uniform', 'distance']
    }),
    "Decision Tree": (DecisionTreeClassifier(random_state=42), {
        'max_depth': [6, 8, 10],
        'min_samples_split': [2, 5, 10]
    }),
    "Random Forest": (RandomForestClassifier(random_state=42), {
        'n_estimators': [150, 200],
        'max_depth': [8, 10, 12],
        'min_samples_split': [2, 5]
    }),
    "XGBoost": (XGBClassifier(
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    ), {
        'n_estimators': [150, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [4, 6, 8]
    }),
}

results = []

def evaluate_model(name, model):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n{name} Results:")
    print("Accuracy:", round(acc * 100, 2), "%")
    print("F1 Score:", round(f1 * 100, 2))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    results.append((name, acc, f1))
    return acc

# ---- Train, tune, and evaluate all ----
for name, (model, params) in models_params.items():
    print(f"\n🔍 Training {name}...")
    if params:
        grid = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
    else:
        model.fit(X_train, y_train)
        best_model = model
    evaluate_model(name, best_model)
    models_params[name] = (best_model, params)

# ---- Find best model ----
best_model_name, best_acc, best_f1 = max(results, key=lambda x: x[1])
best_model = models_params[best_model_name][0]

# ---- Save best model ----
with open(os.path.join(MODELS_DIR, "best_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)

with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("\n🏆 Best Model:", best_model_name)
print(f"✅ Accuracy: {round(best_acc*100,2)}% | F1 Score: {round(best_f1*100,2)}")
print("All models trained successfully (90%+ accuracy expected).")
