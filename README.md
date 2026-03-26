# AutoML Playground

## Overview
AutoML Playground is a Streamlit-based AutoML app for tabular datasets. You can upload a CSV or select a registered benchmark dataset, choose a target column, and let the app:

- detect regression vs classification
- profile the dataset
- shortlist suitable models
- run a champion round with sub-model variants
- run lightweight hyperparameter tuning
- select and save the best model

The UI also includes:

- automated EDA
- training progress feedback
- evaluation charts
- feature importance
- optional SHAP explainability

## Project Structure
```text
automl-project/
├── app.py
├── streamlit_app.py
├── main.py
├── src/
│   └── automl_pipeline.py
├── config/
│   └── dataset_registry.json
├── data/
│   └── data.csv
├── models/
│   └── best_model.pkl
├── requirements.txt
├── Procfile
├── runtime.txt
└── README.md
```

## How Training Works
The pipeline runs in three stages:

1. Stage 1: shortlist models recommended from the dataset profile
2. Stage 2: champion round with tuned sub-model variants
3. Stage 3: lightweight hyperparameter tuning on the strongest candidates

For larger datasets, the app now uses a faster path:

- reduced benchmark shortlist
- fewer CV folds
- lighter default ensemble sizes
- smaller tuning rounds
- sampling for very large datasets

## Dataset Registry
Registered datasets are defined in:

```text
config/dataset_registry.json
```

Each entry stores:

- dataset name
- file path
- task type
- default target column
- cleanup instructions
- benchmark role

## Local Run
Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

Or with the project venv:

```bash
.venv/bin/streamlit run app.py
```

## Deployment
The project includes deployment files for platforms that support a `Procfile` workflow.

Files added for deployment:

- `Procfile`
- `runtime.txt`
- `.gitignore`

Start command:

```text
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

This setup is suitable for platforms such as Render, Railway, and similar services.

## Notes
- `shap` is optional for advanced explainability, but listed in `requirements.txt`
- the saved model is written to `models/best_model.pkl`
- `streamlit_app.py` is a simple wrapper around `app.py`
