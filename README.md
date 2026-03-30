# AutoML Playground

## Overview
AutoML Playground is a Python-based machine learning project that trains multiple models on tabular datasets, compares them, and selects the best fit automatically through a Streamlit interface.

The system supports:
- regression and classification datasets
- automatic preprocessing
- multiple model benchmarking
- cross-validation based model comparison
- explainability and evaluation charts
- local and cloud deployment

## Project Structure
```text
automl-playground/
├── app/
│   ├── __init__.py
│   └── streamlit_ui.py
├── config/
│   └── dataset_registry.json
├── data/
│   └── data.csv
├── models/
│   └── .gitkeep
├── notebooks/
│   └── Untitled5.ipynb
├── reports/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── automl_pipeline.py
│   ├── data_preprocessing.py
│   ├── evaluate.py
│   ├── model_selection.py
│   └── train.py
├── app.py
├── main.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Architecture
The project is split into focused machine learning modules:

- `src/data_preprocessing.py`
  handles data cleaning, task detection, encoding, splitting, and preprocessing pipelines
- `src/model_selection.py`
  defines the candidate model pool and dataset-driven shortlist recommendation logic
- `src/evaluate.py`
  computes ML metrics and extracts feature importance
- `src/train.py`
  provides the training service used by the CLI and the Streamlit app
- `src/automl_pipeline.py`
  orchestrates the AutoML training flow from profiling to best-model selection
- `app/streamlit_ui.py`
  contains the Streamlit frontend

## ML Workflow
1. Load a dataset from the registry or upload a CSV
2. Select the target column
3. Detect or override the task type
4. Build preprocessing steps using `sklearn.Pipeline`
5. Recommend models based on dataset profile
6. Benchmark the shortlisted models with cross-validation
7. Select the best model directly from Stage 1
8. Save the model bundle to `models/best_model.pkl`
9. Display metrics, charts, and explainability in the UI

## Models Used
### Classification
- LogisticRegression
- RandomForestClassifier
- ExtraTreesClassifier
- GradientBoostingClassifier
- HistGradientBoostingClassifier
- SVM
- KNeighborsClassifier
- XGBoostClassifier
- LightGBMClassifier
- CatBoostClassifier

### Regression
- Ridge
- ElasticNet
- RandomForestRegressor
- ExtraTreesRegressor
- GradientBoostingRegressor
- HistGradientBoostingRegressor
- SVR
- KNeighborsRegressor
- XGBoostRegressor
- LightGBMRegressor
- CatBoostRegressor

## Tech Stack
- Python
- Streamlit
- pandas
- numpy
- scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Plotly
- SHAP
- joblib

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## CLI Training
```bash
python main.py
```

## Deployment
The project is prepared for deployment with Streamlit Community Cloud and other similar platforms.

## Notes
- `app.py` is the root entrypoint used for deployment compatibility
- the actual Streamlit implementation lives in `app/streamlit_ui.py`
- the notebook is kept in `notebooks/` as the original experimentation version
