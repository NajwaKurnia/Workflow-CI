import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
import argparse
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# Argparse 
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=10)
parser.add_argument("--test_size", type=float, default=0.2)
args = parser.parse_args()

token = os.environ.get("DAGSHUB_TOKEN", "")
os.environ["MLFLOW_TRACKING_USERNAME"] = token
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

# Setup DagsHub
dagshub.init(
    repo_owner="NajwaKurnia",
    repo_name="Ekperimen_SML_Najwa-Kurnia",
    mlflow=True
)

# Load Data 
df = pd.read_csv("winequality_preprocessing/winequality_clean.csv")
X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=42
)

# MLflow Manual Logging 

with mlflow.start_run(run_name="CI_RandomForest"):

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("test_size", args.test_size)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, model.predict_proba(X_test),
                        multi_class='ovr', average='weighted')
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", roc)

    # Artefak 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # Artefak 2: Feature Importance
    feat_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feat_df, x="importance", y="feature", ax=ax)
    ax.set_title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()

    mlflow.sklearn.log_model(model, "model")
    print(f"Accuracy: {acc:.4f} | ROC-AUC: {roc:.4f}")