"""Étape 2 : nettoie les données brutes et les sépare en jeux train/test (data/processed/)."""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "telco_churn.csv"
OUT_DIR = ROOT / "data" / "processed"


def clean(df):
    # TotalCharges contient des espaces pour les nouveaux clients -> NaN puis 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    # entiers -> float pour que le schéma MLflow accepte d'éventuelles valeurs manquantes
    df[["tenure", "SeniorCitizen"]] = df[["tenure", "SeniorCitizen"]].astype(float)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df.drop(columns=["customerID"])


def main():
    params = yaml.safe_load(open(ROOT / "params.yaml"))["prepare"]
    df = clean(pd.read_csv(RAW))

    train, test = train_test_split(
        df, test_size=params["test_size"], stratify=df["Churn"], random_state=params["random_state"]
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    print(f"train: {len(train)} lignes, test: {len(test)} lignes")


if __name__ == "__main__":
    main()
