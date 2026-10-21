"""Étape 1 : télécharge le dataset brut depuis Kaggle (sans compte) vers data/raw/."""
import shutil
from pathlib import Path

import kagglehub

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "telco_churn.csv"
CSV_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def main():
    src = Path(kagglehub.dataset_download("blastchar/telco-customer-churn")) / CSV_NAME
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, OUT)
    print(f"Données brutes -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
