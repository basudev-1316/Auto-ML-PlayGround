from pathlib import Path

import pandas as pd

from src.automl_pipeline import AutoMLPipeline


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
TARGET_COLUMN = "price"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    automl = AutoMLPipeline(df, target=TARGET_COLUMN, model_path=MODEL_PATH)
    results = automl.run()

    print("\nModel Comparison:")
    print(results)
    print(f"\nBest model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
