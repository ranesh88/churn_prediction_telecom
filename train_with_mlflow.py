import os
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)

import mlflow
import mlflow.sklearn

# ---------------- Base directories ----------------
BASE_DIR = Path(__file__).parent.resolve()
MLRUNS_DIR = BASE_DIR / "mlruns"
MODELS_DIR = BASE_DIR / "models"

MLRUNS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ---------------- MLflow tracking URI (cross-platform) ----------------
mlflow_tracking_uri = MLRUNS_DIR.resolve().as_uri()  # file:// URI
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("Telecom_Churn_Experiment")
print(f"✅ MLflow tracking URI: {mlflow_tracking_uri}")

# ---------------- Load data ----------------
data_path = BASE_DIR / "Churn_Prediction_Final.csv"
data = pd.read_csv(data_path)

# ---------------- Preprocessing ----------------
y = data["Churn"]
X = pd.get_dummies(data.drop("Churn", axis=1), drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Train & log ----------------
with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    class_report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, ROC AUC: {roc_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_path = MODELS_DIR / "confusion_matrix.png"
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(cm_path); plt.close()
    mlflow.log_artifact(str(cm_path))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_path = MODELS_DIR / "roc_curve.png"
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(roc_path); plt.close()
    mlflow.log_artifact(str(roc_path))

    # Save model
    model_path = MODELS_DIR / "rf_classifier.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(str(model_path))

    # Save classification report
    report_path = MODELS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(class_report)
    mlflow.log_artifact(str(report_path))

    # Log params & metrics
    mlflow.log_param("n_estimators", 150)
    mlflow.log_param("max_depth", 8)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)

print("✅ Training complete. Run MLflow UI with: mlflow ui --port 5000")
