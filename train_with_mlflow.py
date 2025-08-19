import os
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)

import mlflow
import mlflow.sklearn

# ---------------- MLflow authentication ----------------
dagshub_username = "ranesh88"
dagshub_token = "1034242442f1c28d28868dae1e2701a2df4e0d35"
mlflow_tracking_uri = f"https://{dagshub_username}:{dagshub_token}@dagshub.com/ranesh88/churn_prediction_telecom.mlflow"
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("Telecom_Churn_Experiment")
print(f"✅ MLflow tracking URI: {mlflow_tracking_uri}")

# ---------------- Base directories ----------------
BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

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
    avg_precision = average_precision_score(y_test, y_proba)
    class_report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, ROC AUC: {roc_auc:.4f}, PR-AUC: {avg_precision:.4f}")

    # ---------------- Confusion Matrix ----------------
    cm = confusion_matrix(y_test, y_pred)
    cm_path = MODELS_DIR / "confusion_matrix.png"
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(cm_path); plt.close()
    mlflow.log_artifact(str(cm_path))

    # ---------------- ROC Curve ----------------
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_path = MODELS_DIR / "roc_curve.png"
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})", color="darkorange", linewidth=2)
    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(roc_path); plt.close()
    mlflow.log_artifact(str(roc_path))

    # ---------------- Precision-Recall Curve ----------------
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_path = MODELS_DIR / "pr_curve.png"
    plt.figure()
    plt.plot(recall_vals, precision_vals, color="purple", linewidth=2, label=f"AP = {avg_precision:.2f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left"); plt.tight_layout()
    plt.savefig(pr_path); plt.close()
    mlflow.log_artifact(str(pr_path))

    # ---------------- Feature Importance ----------------
    importances = model.feature_importances_
    feature_names = X.columns
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)

    fi_path = MODELS_DIR / "feature_importance.png"
    plt.figure(figsize=(8,5))
    sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis")
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout(); plt.savefig(fi_path); plt.close()
    mlflow.log_artifact(str(fi_path))

    # ---------------- Classification Report ----------------
    report_path = MODELS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(class_report)
    mlflow.log_artifact(str(report_path))

    # ---------------- Hypothesis Testing (Chi-Square) ----------------
    chi2_results = []
    for col in data.select_dtypes(include='object').columns:
        if col != 'Churn':
            ct = pd.crosstab(data[col], data['Churn'])
            chi2, p, dof, _ = chi2_contingency(ct)
            chi2_results.append({
                'Feature': col,
                'Chi2 Statistic': round(chi2, 3),
                'p-value': round(p, 4),
                'Degrees of Freedom': dof,
                'Significant': 'Yes' if p < 0.05 else 'No'
            })

    chi2_df = pd.DataFrame(chi2_results)
    chi2_path = MODELS_DIR / "chi_square_results.csv"
    chi2_df.to_csv(chi2_path, index=False)
    mlflow.log_artifact(str(chi2_path))
    print("✅ Chi-Square Hypothesis Testing results saved and logged.")

    # ---------------- Log parameters & metrics ----------------
    mlflow.log_param("n_estimators", 150)
    mlflow.log_param("max_depth", 8)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("pr_auc", avg_precision)

print("✅ Training complete. All metrics, plots, and hypothesis testing logged directly to DagsHub MLflow.")
