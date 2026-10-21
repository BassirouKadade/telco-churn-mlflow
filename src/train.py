"""
Étape 3 : entraîne et compare plusieurs modèles, avec suivi MLflow.

Chaque modèle entraîné = un "run" MLflow, avec ses paramètres, ses métriques,
sa matrice de confusion, le modèle lui-même et l'empreinte des données utilisées
(le même hash md5 que DVC, pour relier chaque run à une version précise des données).
Les scores sont aussi écrits dans metrics.json pour `dvc metrics show`.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas de fenêtre graphique, on enregistre juste les figures
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
MODEL_CLASSES = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def load(name):
    df = pd.read_csv(DATA / name)
    y = df.pop("Churn")
    return df, y


def build_pipeline(model, X):
    categorical = [c for c in X.columns if c not in NUMERIC]
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("preprocess", preprocess), ("model", model)])


def make_model(name, hyperparams, random_state):
    cls = MODEL_CLASSES[name]
    if "random_state" in cls().get_params():
        hyperparams = {**hyperparams, "random_state": random_state}
    return cls(**hyperparams)


def main():
    params = yaml.safe_load(open(ROOT / "params.yaml"))["train"]
    X_train, y_train = load("train.csv")
    X_test, y_test = load("test.csv")
    data_tags = {"data.train_md5": md5(DATA / "train.csv"), "data.test_md5": md5(DATA / "test.csv")}

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("telco-churn")

    all_metrics = {}
    for name, hyperparams in params["models"].items():
        model = make_model(name, hyperparams, params["random_state"])
        with mlflow.start_run(run_name=name):
            pipe = build_pipeline(model, X_train)
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]

            # 1) Paramètres : ce qu'on a choisi
            mlflow.log_param("model_type", name)
            mlflow.log_params(model.get_params())
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_test", len(X_test))
            mlflow.set_tags(data_tags)

            # 2) Métriques : ce qu'on a obtenu
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }
            mlflow.log_metrics(metrics)
            all_metrics[name] = {k: round(v, 4) for k, v in metrics.items()}

            # 3) Artefact : la matrice de confusion en image
            fig, ax = plt.subplots(figsize=(4, 4))
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, colorbar=False)
            ax.set_title(name)
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)

            # 4) Le modèle lui-même (pipeline complet, prêt à être rechargé/déployé)
            signature = infer_signature(X_test, y_pred)
            # skops refuse par défaut de recharger les arbres ; on les a créés nous-mêmes, donc on les autorise
            mlflow.sklearn.log_model(
                pipe, name="model", signature=signature, input_example=X_test.head(3),
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            print(f"{name:22s} " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    (ROOT / "metrics.json").write_text(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
