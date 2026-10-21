"""
Prédiction du churn client (dataset Kaggle "Telco Customer Churn") avec suivi MLflow.

Chaque modèle entraîné = un "run" MLflow, avec ses paramètres, ses métriques,
sa matrice de confusion et le modèle lui-même sauvegardé.

Usage :
    python train.py                       # entraîne les 3 modèles (télécharge le CSV si besoin)
    python train.py --data mon_fichier.csv
Puis :
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas de fenêtre graphique, on enregistre juste les figures
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).parent
CSV_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DEFAULT_DATA = HERE / "data" / CSV_NAME


def get_data_path(path):
    """Renvoie le chemin du CSV ; s'il est absent, le télécharge depuis Kaggle (sans compte)."""
    if Path(path).exists():
        return path
    import kagglehub
    print("CSV introuvable, téléchargement depuis Kaggle...")
    return Path(kagglehub.dataset_download("blastchar/telco-customer-churn")) / CSV_NAME


def load_data(path):
    df = pd.read_csv(path)
    # TotalCharges contient des espaces pour les nouveaux clients -> NaN puis 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    # entiers -> float pour que le schéma MLflow accepte d'éventuelles valeurs manquantes
    df[["tenure", "SeniorCitizen"]] = df[["tenure", "SeniorCitizen"]].astype(float)
    df = df.drop(columns=["customerID"])
    y = (df.pop("Churn") == "Yes").astype(int)
    return df, y


def build_pipeline(model, X):
    numeric = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    categorical = [c for c in X.columns if c not in numeric]
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("preprocess", preprocess), ("model", model)])


# Les modèles à comparer (leurs hyperparamètres seront loggés automatiquement)
MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, C=1.0),
    "random_forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    args = parser.parse_args()

    X, y = load_data(get_data_path(args.data))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    mlflow.set_tracking_uri(f"sqlite:///{HERE / 'mlflow.db'}")
    mlflow.set_experiment("telco-churn")

    for name, model in MODELS.items():
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

            # 2) Métriques : ce qu'on a obtenu
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }
            mlflow.log_metrics(metrics)

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


if __name__ == "__main__":
    main()
