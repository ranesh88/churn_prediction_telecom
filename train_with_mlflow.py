import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- MLflow / Dagshub Setup ----------------
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")
DAGSHUB_USER_TOKEN = os.getenv("DAGSHUB_USER_TOKEN")

if DAGSHUB_USERNAME and DAGSHUB_REPO and DAGSHUB_USER_TOKEN:
    # GitHub Actions / CI environment
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
    print("✅ MLflow tracking set to Dagshub remote")
else:
    # Local environment
    mlflow.set_tracking_uri("mlruns")
    print("✅ MLflow tracking set to local 'mlruns' folder")

mlflow.set_experiment("Telecom_Churn_Experiment")

# ---------------- Load dataset ----------------
df = pd.read_csv("Churn_Prediction_Final.csv")

# ---------------- Preprocessing ----------------
y = df["Churn"]
X = df.drop("Churn", axis=1)
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Training ----------------
with mlflow.start_run():
    print("Training RandomForest model...")
    n_estimators = 100
    random_state = 42

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
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

    # Console output
    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1-score   : {f1:.4f}")
    print(f"ROC AUC    : {roc_auc:.4f}")
    print("\n=== Classification Report ===")
    print(class_report)

    # ---------------- Confusion Matrix ----------------
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
    print("\n=== Confusion Matrix ===")
    print(cm_df)

    # Save confusion matrix plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    os.makedirs("models", exist_ok=True)
    cm_path = os.path.join("models", "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    roc_path = os.path.join("models", "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    mlflow.log_artifact(roc_path)

    # ---------------- Save model ----------------
    model_path = os.path.join("models", "rf_classifier.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    mlflow.log_artifact(model_path, artifact_path="models")

    # ---------------- Log parameters & metrics ----------------
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)

    # Save classification report
    report_path = os.path.join("models", "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("=== Classification Report ===\n")
        f.write(class_report)
    mlflow.log_artifact(report_path)

print("\nRun MLflow UI locally with: mlflow ui --port 5000")
