from pathlib import Path

import pandas as pd

from src.train import run_training


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
TARGET_COLUMN = "price"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    results, evaluation_artifacts = run_training(
        df=df,
        target_column=TARGET_COLUMN,
        model_path=MODEL_PATH,
        training_mode="Balanced",
    )

    print("\nModel Comparison:")
    print(results)
    print(f"\nDetected task: {evaluation_artifacts.get('task_type')}")
    print(f"Best model: {evaluation_artifacts.get('best_model_name')}")
    print(f"\nBest model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
