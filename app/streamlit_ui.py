from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from src.train import run_training


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
REGISTRY_PATH = PROJECT_ROOT / "config" / "dataset_registry.json"


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_dataset_registry() -> dict[str, object]:
    with REGISTRY_PATH.open() as registry_file:
        return json.load(registry_file)


@st.cache_data
def load_registered_dataset(dataset_path: str) -> pd.DataFrame:
    resolved_path = Path(dataset_path)
    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path
    return pd.read_csv(resolved_path)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        return {
            "load_error": str(exc),
            "model_path": str(MODEL_PATH),
        }


def train_model(
    df: pd.DataFrame,
    target_column: str,
    task_type_override: str | None = None,
    training_mode: str = "Balanced",
    progress_callback=None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    results, evaluation_artifacts = run_training(
        df=df,
        target_column=target_column,
        model_path=MODEL_PATH,
        task_type_override=task_type_override,
        training_mode=training_mode,
        progress_callback=progress_callback,
    )
    load_model.clear()
    return results, evaluation_artifacts


def render_training_progress(payload: dict[str, object], status_box, progress_bar) -> None:
    phase = payload.get("phase", "training")
    message = payload.get("message", "Working...")
    current = payload.get("current")
    total = payload.get("total")

    phase_labels = {
        "profiling": "Profiling Dataset",
        "shortlist": "Model Benchmarking",
        "finalizing": "Finalizing Winner",
        "saving": "Saving Model",
        "done": "Completed",
    }
    label = phase_labels.get(phase, "Training")
    status_box.info(f"{label}: {message}")

    phase_progress = {
        "profiling": 0.08,
        "shortlist": 0.78,
        "finalizing": 0.94,
        "saving": 0.97,
        "done": 1.0,
    }
    progress_value = phase_progress.get(phase, 0.1)

    if current is not None and total:
        if phase == "shortlist":
            progress_value = 0.10 + (current / total) * 0.72

    progress_bar.progress(min(max(progress_value, 0.0), 1.0))


def render_dataset_plots(dataset: pd.DataFrame, target_column: str) -> None:
    numeric_columns = dataset.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = dataset.select_dtypes(exclude=["number"]).columns.tolist()

    overview_col, quality_col, stats_col = st.columns(3)
    with overview_col:
        st.metric("Rows", f"{len(dataset):,}")
    with quality_col:
        st.metric("Columns", len(dataset.columns))
    with stats_col:
        st.metric("Missing Cells", int(dataset.isna().sum().sum()))

    info_col_1, info_col_2, info_col_3 = st.columns(3)
    with info_col_1:
        st.metric("Numeric Columns", len(numeric_columns))
    with info_col_2:
        st.metric("Categorical Columns", len(categorical_columns))
    with info_col_3:
        unique_target = int(dataset[target_column].nunique(dropna=True)) if target_column in dataset.columns else 0
        st.metric("Target Classes/Values", unique_target)

    st.subheader("Automated EDA")

    missing_counts = dataset.isna().sum()
    if missing_counts.sum() > 0:
        missing_df = missing_counts[missing_counts > 0].sort_values(ascending=False).reset_index()
        missing_df.columns = ["column", "missing_values"]
        st.plotly_chart(
            px.bar(
                missing_df,
                x="column",
                y="missing_values",
                title="Missing Values by Column",
            ),
            use_container_width=True,
        )
    else:
        st.info("No missing values found in the uploaded dataset.")

    if target_column in numeric_columns:
        st.plotly_chart(
            px.histogram(
                dataset,
                x=target_column,
                nbins=30,
                title=f"Target Distribution: {target_column}",
            ),
            use_container_width=True,
        )
    else:
        target_counts = (
            dataset[target_column]
            .astype(str)
            .value_counts(dropna=False)
            .head(20)
            .reset_index()
        )
        target_counts.columns = ["target_value", "count"]
        st.plotly_chart(
            px.bar(
                target_counts,
                x="target_value",
                y="count",
                title=f"Target Class Balance: {target_column}",
            ),
            use_container_width=True,
        )

    if numeric_columns:
        numeric_preview = numeric_columns[: min(6, len(numeric_columns))]
        numeric_long_df = dataset[numeric_preview].melt(var_name="feature", value_name="value").dropna()
        if not numeric_long_df.empty:
            st.plotly_chart(
                px.histogram(
                    numeric_long_df,
                    x="value",
                    facet_col="feature",
                    facet_col_wrap=3,
                    nbins=25,
                    title="Numeric Feature Distributions",
                ),
                use_container_width=True,
            )

    if len(numeric_columns) >= 2:
        correlation_df = dataset[numeric_columns].corr(numeric_only=True)
        st.plotly_chart(
            px.imshow(
                correlation_df,
                text_auto=".2f",
                aspect="auto",
                title="Correlation Heatmap",
            ),
            use_container_width=True,
        )

        if target_column in correlation_df.columns:
            target_correlations = (
                correlation_df[target_column]
                .drop(labels=[target_column], errors="ignore")
                .abs()
                .sort_values(ascending=False)
                .head(8)
                .reset_index()
            )
            target_correlations.columns = ["feature", "absolute_correlation"]
            st.plotly_chart(
                px.bar(
                    target_correlations,
                    x="feature",
                    y="absolute_correlation",
                    title=f"Top Numeric Relationships with {target_column}",
                ),
                use_container_width=True,
            )


def render_eda_section(dataset: pd.DataFrame, target_column: str) -> None:
    st.markdown("## Dataset Explorer")
    with st.expander("Open Automated EDA", expanded=False):
        render_dataset_plots(dataset, target_column)


def render_model_results(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        return

    chart_df = results_df.copy()
    metric_name = str(chart_df["metric_name"].iloc[0]) if "metric_name" in chart_df.columns and not chart_df.empty else "Score"
    ascending = metric_name.upper() == "RMSE"
    chart_df["is_best"] = chart_df["best"].astype(str).str.len() > 0
    chart_df["label"] = chart_df["is_best"].map({True: "Best Model", False: "Other Models"})

    best_rows = chart_df[chart_df["is_best"]]
    if not best_rows.empty:
        best_model_name = str(best_rows.iloc[0]["model"])
        best_cv_score = float(best_rows.iloc[0]["cv_score"])
        best_test_score = float(best_rows.iloc[0]["test_score"])
        st.subheader("Final Winner")
        metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
        with metric_col_1:
            st.metric("Best Model", best_model_name)
        with metric_col_2:
            st.metric("Best CV Score", f"{best_cv_score:.4f}")
        with metric_col_3:
            st.metric("Best Test Score", f"{best_test_score:.4f}")
        st.success(f"Best fit model: {best_model_name} with CV score {best_cv_score:.4f}")

    if "stage" not in chart_df.columns:
        st.subheader("Model Performance")
        st.dataframe(results_df, use_container_width=True)
        fig = px.bar(
            chart_df.sort_values("cv_score", ascending=ascending),
            x="model",
            y="cv_score",
            color="label",
            text="cv_score",
            title=f"Cross-Validation {metric_name} by Model",
            color_discrete_map={
                "Best Model": "#1f77b4",
                "Other Models": "#bfc7d5",
            },
        )
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(yaxis_title=metric_name, xaxis_title="Model")
        st.plotly_chart(fig, use_container_width=True)
        return

    shortlist_df = chart_df[chart_df["stage"] == "shortlist"].copy()

    st.subheader("Model Benchmark Results")
    if shortlist_df.empty:
        st.info("No shortlist-stage results available.")
    else:
        st.dataframe(
            shortlist_df[["stage", "model", "cv_score", "test_score", "best"]].sort_values("cv_score", ascending=ascending),
            use_container_width=True,
        )
        shortlist_fig = px.bar(
            shortlist_df.sort_values("cv_score", ascending=ascending),
            x="model",
            y="cv_score",
            color="label",
            text="cv_score",
            title=f"Shortlist Round {metric_name}",
            color_discrete_map={
                "Best Model": "#1f77b4",
                "Other Models": "#bfc7d5",
            },
        )
        shortlist_fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        shortlist_fig.update_layout(yaxis_title=metric_name, xaxis_title="Model")
        st.plotly_chart(shortlist_fig, use_container_width=True)


def render_evaluation_panel(evaluation_artifacts: dict[str, object]) -> None:
    task_type = evaluation_artifacts.get("task_type")
    y_test = evaluation_artifacts.get("y_test")
    predictions = evaluation_artifacts.get("predictions")
    probabilities = evaluation_artifacts.get("probabilities")
    metrics = evaluation_artifacts.get("metrics") or {}
    best_model_name = evaluation_artifacts.get("best_model_name", "Best Model")

    if y_test is None or predictions is None:
        return

    st.subheader("Best Model Evaluation")
    st.caption(f"Evaluation charts for {best_model_name}")

    if task_type == "classification":
        metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
        with metric_col_1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0.0):.4f}")
        with metric_col_2:
            st.metric("Precision", f"{metrics.get('precision', 0.0):.4f}")
        with metric_col_3:
            st.metric("Recall", f"{metrics.get('recall', 0.0):.4f}")
        with metric_col_4:
            st.metric("F1 Score", f"{metrics.get('f1', 0.0):.4f}")

        roc_auc = metrics.get("roc_auc")
        if roc_auc is not None:
            st.metric("ROC-AUC", f"{roc_auc:.4f}")

        labels = sorted(pd.Series(y_test).dropna().unique().tolist())
        cm = confusion_matrix(y_test, predictions, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"Actual {label}" for label in labels], columns=[f"Pred {label}" for label in labels])
        st.plotly_chart(
            px.imshow(
                cm_df,
                text_auto=True,
                title="Confusion Matrix",
                aspect="auto",
            ),
            use_container_width=True,
        )

        unique_labels = pd.Series(y_test).dropna().unique().tolist()
        if len(unique_labels) == 2 and probabilities is not None:
            try:
                if getattr(probabilities, "ndim", 1) == 2 and probabilities.shape[1] >= 2:
                    positive_scores = probabilities[:, 1]
                else:
                    positive_scores = probabilities

                positive_label = labels[-1]
                binary_truth = (pd.Series(y_test) == positive_label).astype(int)

                roc_fpr, roc_tpr, _ = roc_curve(binary_truth, positive_scores)
                pr_precision, pr_recall, _ = precision_recall_curve(binary_truth, positive_scores)

                roc_col, pr_col = st.columns(2)
                with roc_col:
                    roc_fig = go.Figure()
                    roc_fig.add_trace(
                        go.Scatter(
                            x=roc_fpr,
                            y=roc_tpr,
                            mode="lines",
                            name="ROC Curve",
                            line={"color": "#29b6f6", "width": 3},
                        )
                    )
                    roc_fig.add_trace(
                        go.Scatter(
                            x=[0, 1],
                            y=[0, 1],
                            mode="lines",
                            name="Baseline",
                            line={"color": "#8b97aa", "dash": "dash"},
                        )
                    )
                    roc_fig.update_layout(
                        title="ROC Curve",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                    )
                    st.plotly_chart(roc_fig, use_container_width=True)

                with pr_col:
                    pr_fig = go.Figure()
                    pr_fig.add_trace(
                        go.Scatter(
                            x=pr_recall,
                            y=pr_precision,
                            mode="lines",
                            name="Precision-Recall",
                            line={"color": "#ff7a18", "width": 3},
                        )
                    )
                    pr_fig.update_layout(
                        title="Precision-Recall Curve",
                        xaxis_title="Recall",
                        yaxis_title="Precision",
                    )
                    st.plotly_chart(pr_fig, use_container_width=True)
            except Exception:
                pass
        return

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    with metric_col_1:
        st.metric("R2", f"{metrics.get('r2', 0.0):.4f}")
    with metric_col_2:
        st.metric("MAE", f"{metrics.get('mae', 0.0):,.2f}")
    with metric_col_3:
        st.metric("RMSE", f"{metrics.get('rmse', 0.0):,.2f}")

    actual_vs_pred = pd.DataFrame({"actual": y_test, "predicted": predictions})
    st.plotly_chart(
        px.scatter(
            actual_vs_pred,
            x="actual",
            y="predicted",
            title="Actual vs Predicted",
        ),
        use_container_width=True,
    )

    residuals_df = actual_vs_pred.copy()
    residuals_df["residual"] = residuals_df["actual"] - residuals_df["predicted"]
    residuals_fig = go.Figure()
    residuals_fig.add_trace(
        go.Scatter(
            x=residuals_df["predicted"],
            y=residuals_df["residual"],
            mode="markers",
            name="Residuals",
        )
    )
    residuals_fig.add_hline(y=0, line_dash="dash")
    residuals_fig.update_layout(
        title="Residual Plot",
        xaxis_title="Predicted",
        yaxis_title="Residual",
    )
    st.plotly_chart(residuals_fig, use_container_width=True)


def build_shap_summary(model: object, x_test: pd.DataFrame) -> pd.DataFrame | None:
    try:
        import shap
    except Exception:
        return None

    if model is None or x_test is None or x_test.empty:
        return None

    if not hasattr(model, "named_steps"):
        return None

    try:
        preprocessor = model.named_steps["preprocessor"]
        estimator = model.named_steps["model"]
        sample_size = min(150, len(x_test))
        sample_df = x_test.sample(n=sample_size, random_state=42) if len(x_test) > sample_size else x_test.copy()
        transformed = preprocessor.transform(sample_df)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        feature_names = preprocessor.get_feature_names_out()
        explainer = shap.Explainer(estimator, transformed, feature_names=feature_names)
        shap_values = explainer(transformed)
        shap_array = np.asarray(shap_values.values)

        if shap_array.ndim == 3:
            importance_values = np.mean(np.abs(shap_array), axis=(0, 2))
        elif shap_array.ndim == 2:
            importance_values = np.mean(np.abs(shap_array), axis=0)
        else:
            return None

        summary_df = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": importance_values,
            }
        )
        summary_df = summary_df.sort_values("mean_abs_shap", ascending=False).head(20).reset_index(drop=True)
        return summary_df
    except Exception:
        return None


def render_explainability_panel(evaluation_artifacts: dict[str, object]) -> None:
    best_model_name = evaluation_artifacts.get("best_model_name", "Best Model")
    shap_summary = evaluation_artifacts.get("shap_summary")
    if shap_summary is None:
        shap_summary = build_shap_summary(evaluation_artifacts.get("model"), evaluation_artifacts.get("x_test"))

    if shap_summary is None or shap_summary.empty:
        st.info("Advanced SHAP explainability is available when the `shap` package is installed and supported by the selected best model.")
        return

    st.subheader("SHAP Explainability")
    st.caption(f"Mean absolute SHAP contribution for {best_model_name}")
    st.plotly_chart(
        px.bar(
            shap_summary.sort_values("mean_abs_shap", ascending=True),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title="Top SHAP Feature Contributions",
        ),
        use_container_width=True,
    )
    st.dataframe(shap_summary, use_container_width=True)


def render_feature_importance_panel(evaluation_artifacts: dict[str, object]) -> None:
    feature_importance = evaluation_artifacts.get("feature_importance")
    best_model_name = evaluation_artifacts.get("best_model_name", "Best Model")

    if feature_importance is None:
        st.info("Feature importance is not available for the selected best model.")
        return

    if isinstance(feature_importance, pd.DataFrame):
        importance_df = feature_importance.copy()
    else:
        importance_df = pd.DataFrame(feature_importance)

    if importance_df.empty or "feature" not in importance_df.columns or "importance" not in importance_df.columns:
        st.info("Feature importance is not available for the selected best model.")
        return

    st.subheader("Feature Importance")
    st.caption(f"Top contributing features for {best_model_name}")
    st.plotly_chart(
        px.bar(
            importance_df.sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            title="Top Feature Contributions",
        ),
        use_container_width=True,
    )
    st.dataframe(importance_df, use_container_width=True)


def render_dataset_profile_panel(evaluation_artifacts: dict[str, object]) -> None:
    dataset_profile = evaluation_artifacts.get("dataset_profile")
    if dataset_profile is None:
        st.info("Train the model to see dataset-driven model recommendations.")
        return

    st.subheader("Dataset Profile")
    profile_col_1, profile_col_2, profile_col_3, profile_col_4 = st.columns(4)
    with profile_col_1:
        st.metric("Rows", f"{dataset_profile.row_count:,}")
    with profile_col_2:
        st.metric("Features", dataset_profile.column_count)
    with profile_col_3:
        st.metric("Numeric Features", dataset_profile.numeric_feature_count)
    with profile_col_4:
        st.metric("Categorical Features", dataset_profile.categorical_feature_count)

    info_col_1, info_col_2 = st.columns(2)
    with info_col_1:
        st.write(f"Task type: `{dataset_profile.task_type}`")
        st.write(f"Missing ratio: `{dataset_profile.missing_ratio:.2%}`")
    with info_col_2:
        if dataset_profile.high_cardinality_columns:
            st.write("High-cardinality columns:")
            st.write(", ".join(dataset_profile.high_cardinality_columns[:5]))
        else:
            st.write("High-cardinality columns: none detected")

    st.write("Why these models were selected:")
    for reason in dataset_profile.reasoning:
        st.write(f"- {reason}")

    recommendation_df = pd.DataFrame(
        {
            "recommended_model": dataset_profile.recommended_models,
        }
    )
    st.dataframe(recommendation_df, use_container_width=True)


def prepare_registered_dataset(dataset_entry: dict[str, object]) -> pd.DataFrame:
    dataset = load_registered_dataset(str(dataset_entry["path"])).copy()

    for column in dataset_entry.get("drop_columns", []):
        if column in dataset.columns:
            dataset = dataset.drop(columns=[column])

    for column, action in dataset_entry.get("drop_or_convert_columns", {}).items():
        if column not in dataset.columns:
            continue
        if action == "convert_to_numeric":
            dataset[column] = pd.to_numeric(dataset[column], errors="coerce")

    return dataset


def registry_dataset_exists(dataset_entry: dict[str, object]) -> bool:
    dataset_path = Path(str(dataset_entry["path"]))
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    return dataset_path.exists()


def inject_custom_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --aml-bg-start: #08111f;
            --aml-bg-end: #13233d;
            --aml-accent: #ff7a18;
            --aml-accent-2: #29b6f6;
            --aml-card: rgba(13, 20, 35, 0.82);
            --aml-border: rgba(255, 255, 255, 0.08);
            --aml-text: #f5f7fb;
            --aml-muted: #b7c3d8;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255,122,24,0.18), transparent 24%),
                radial-gradient(circle at top right, rgba(41,182,246,0.16), transparent 20%),
                linear-gradient(160deg, var(--aml-bg-start) 0%, var(--aml-bg-end) 100%);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(8, 14, 28, 0.98), rgba(16, 24, 42, 0.96));
            border-right: 1px solid var(--aml-border);
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: var(--aml-text) !important;
        }

        .hero-shell {
            padding: 1.2rem 0 1.4rem 0;
            margin-bottom: 0.6rem;
        }

        .hero-kicker {
            display: inline-block;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: white;
            background: linear-gradient(90deg, var(--aml-accent), var(--aml-accent-2));
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
            animation: floatBadge 3.5s ease-in-out infinite;
        }

        .hero-title {
            margin: 0.9rem 0 0.25rem 0;
            font-size: clamp(2.8rem, 6vw, 4.8rem);
            line-height: 0.96;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: var(--aml-text);
            text-shadow: 0 8px 30px rgba(0, 0, 0, 0.28);
        }

        .hero-title .gradient {
            background: linear-gradient(90deg, #ffb36b 0%, #ff7a18 38%, #7ad7ff 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            background-size: 200% 200%;
            animation: shiftGradient 6s ease infinite;
        }

        .hero-subtitle {
            margin: 0.5rem 0 0 0;
            color: var(--aml-muted);
            font-size: 1rem;
            max-width: 44rem;
        }

        .hero-line {
            width: 140px;
            height: 4px;
            border-radius: 999px;
            margin-top: 1rem;
            background: linear-gradient(90deg, var(--aml-accent), var(--aml-accent-2));
            background-size: 200% 200%;
            animation: shiftGradient 5s ease infinite;
        }

        .control-shell {
            padding: 1.4rem;
            border-radius: 28px;
            background: linear-gradient(180deg, rgba(10, 16, 30, 0.88), rgba(13, 20, 35, 0.76));
            border: 1px solid var(--aml-border);
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.22);
            backdrop-filter: blur(14px);
            margin-bottom: 1.5rem;
        }

        .control-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--aml-text);
            margin-bottom: 0.25rem;
        }

        .control-subtitle {
            color: var(--aml-muted);
            margin-bottom: 1rem;
        }

        div[data-testid="stButton"] > button {
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.12);
            background: linear-gradient(90deg, rgba(255,122,24,0.18), rgba(41,182,246,0.18));
            color: var(--aml-text);
            font-weight: 700;
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        }

        div[data-testid="stButton"] > button:hover {
            transform: translateY(-1px);
            border-color: rgba(255,255,255,0.26);
            box-shadow: 0 10px 24px rgba(0,0,0,0.24);
        }

        div[data-testid="stDataFrame"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-testid="stNumberInput"] > div,
        div[data-testid="stTabs"] {
            border-radius: 18px;
        }

        @keyframes shiftGradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes floatBadge {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-3px); }
            100% { transform: translateY(0px); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_model_schema_status(
    dataset: pd.DataFrame,
    selected_target: str,
    model_bundle: object,
) -> dict[str, object]:
    status = {
        "trained_model": None,
        "trained_target": selected_target,
        "prediction_features": dataset.drop(columns=[selected_target]),
        "schema_matches": False,
        "missing_columns": [],
        "extra_columns": [],
        "load_error": None,
    }

    if isinstance(model_bundle, dict) and "load_error" in model_bundle:
        status["load_error"] = model_bundle["load_error"]
        return status

    if not isinstance(model_bundle, dict) or "model" not in model_bundle:
        status["trained_model"] = model_bundle
        return status

    trained_model = model_bundle["model"]
    trained_target = model_bundle.get("target", selected_target)
    feature_columns = model_bundle.get("feature_columns") or []

    dataset_feature_columns = [column for column in dataset.columns if column != selected_target]
    missing_columns = [column for column in feature_columns if column not in dataset_feature_columns]
    extra_columns = [column for column in dataset_feature_columns if column not in feature_columns]

    prediction_features = dataset[feature_columns].copy() if feature_columns and not missing_columns else pd.DataFrame(columns=feature_columns)

    status.update(
        {
            "trained_model": trained_model,
            "trained_target": trained_target,
            "prediction_features": prediction_features,
            "schema_matches": len(missing_columns) == 0 and trained_target == selected_target,
            "missing_columns": missing_columns,
            "extra_columns": extra_columns,
        }
    )
    return status


def build_dataset_signature(dataset: pd.DataFrame, target_column: str) -> dict[str, object]:
    feature_columns = [column for column in dataset.columns if column != target_column]
    return {
        "columns": dataset.columns.tolist(),
        "feature_columns": feature_columns,
        "target": target_column,
        "row_count": int(len(dataset)),
    }


def main() -> None:
    st.set_page_config(page_title="AutoML Playground", page_icon="ML", layout="wide")
    inject_custom_styles()
    st.markdown(
        """
        <section class="hero-shell">
            <div class="hero-kicker">Smart Model Discovery</div>
            <h1 class="hero-title">Auto<span class="gradient">ML Playground</span></h1>
            <p class="hero-subtitle">
                Upload a dataset or pick one from the registry, then let the production AutoML
                pipeline preprocess, tune, compare, and explain the best-fit model for your data.
            </p>
            <div class="hero-line"></div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    dataset_registry = load_dataset_registry()
    registry_entries = dataset_registry.get("datasets", [])
    available_registry_entries = [entry for entry in registry_entries if registry_dataset_exists(entry)]
    registry_lookup = {entry["name"]: entry for entry in registry_entries}
    available_registry_lookup = {entry["name"]: entry for entry in available_registry_entries}
    registry_dataset_names = sorted(registry_lookup.keys())
    available_registry_names = sorted(available_registry_lookup.keys())

    st.markdown(
        """
        <section class="control-shell">
            <div class="control-title">Model Setup</div>
            <div class="control-subtitle">Choose a dataset source, configure the task, and launch end-to-end AutoML training.</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    control_left, control_right = st.columns([1.2, 1])
    with control_left:
        default_source_index = 0 if available_registry_names else 1
        dataset_source = st.radio("Dataset source", options=["Registry dataset", "Upload CSV"], index=default_source_index, horizontal=True)
    with control_right:
        uploaded_dataset = None
        selected_registry_name = None
        selected_registry_entry = None

    if dataset_source == "Registry dataset":
        if not available_registry_names:
            st.info("Registry datasets are not bundled in this deployment. Please use Upload CSV.")
            dataset_source = "Upload CSV"
        else:
            selected_registry_name = st.selectbox("Choose dataset", options=available_registry_names)
            selected_registry_entry = available_registry_lookup[selected_registry_name]
    else:
        uploaded_dataset = st.file_uploader("Upload training dataset", type=["csv"])

    if dataset_source == "Registry dataset" and selected_registry_entry is not None:
        dataset = prepare_registered_dataset(selected_registry_entry)
        default_target = selected_registry_entry.get("default_target")
    elif uploaded_dataset is not None:
        dataset = pd.read_csv(uploaded_dataset)
        default_target = "price" if "price" in dataset.columns else None
    else:
        dataset = None
        default_target = None

    if dataset is None:
        st.info("Upload a CSV or choose a registry dataset to begin.")
        return

    if default_target in dataset.columns:
        default_target_index = dataset.columns.get_loc(default_target)
    else:
        default_target_index = max(len(dataset.columns) - 1, 0)

    target_col, task_col, mode_col = st.columns(3)
    with target_col:
        selected_target = st.selectbox("Target column", options=dataset.columns.tolist(), index=default_target_index)
    with task_col:
        task_type_choice = st.selectbox(
            "Task type",
            options=["Auto Detect", "Classification", "Regression"],
            index=0,
        )
    with mode_col:
        training_mode = st.selectbox(
            "Training mode",
            options=["Fast", "Balanced", "Full"],
            index=1,
        )

    task_type_override = None
    if task_type_choice == "Classification":
        task_type_override = "classification"
    elif task_type_choice == "Regression":
        task_type_override = "regression"

    dataset_features = dataset.drop(columns=[selected_target])
    model_bundle = load_model()
    trained_model = None
    trained_target = selected_target
    prediction_features = dataset_features
    training_results_df = None
    evaluation_artifacts = None
    schema_status = get_model_schema_status(dataset, selected_target, model_bundle)
    trained_model = schema_status["trained_model"]
    trained_target = schema_status["trained_target"]
    prediction_features = schema_status["prediction_features"] if not schema_status["prediction_features"].empty else dataset_features
    active_signature = build_dataset_signature(dataset, selected_target)
    saved_signature = None
    if isinstance(model_bundle, dict):
        saved_signature = model_bundle.get("dataset_signature")
    dataset_is_trained = saved_signature == active_signature and schema_status["schema_matches"]

    if schema_status.get("load_error"):
        st.warning(
            "The saved model could not be loaded in this environment. "
            "Retrain the dataset to create a fresh compatible model."
        )
        with st.expander("Saved model load error", expanded=False):
            st.code(str(schema_status["load_error"]))

    render_eda_section(dataset, selected_target)

    if st.button("Train / Retrain Model", use_container_width=True):
        status_box = st.empty()
        progress_bar = st.progress(0.0)
        with st.spinner("Training model..."):
            def progress_callback(payload: dict[str, object]) -> None:
                render_training_progress(payload, status_box, progress_bar)

            training_results_df, evaluation_artifacts = train_model(
                dataset,
                selected_target,
                task_type_override=task_type_override,
                training_mode=training_mode,
                progress_callback=progress_callback,
            )
        st.success(f"Training complete. Model saved to `{MODEL_PATH.name}`.")
        status_box.empty()
        progress_bar.empty()
        model_bundle = load_model()
        schema_status = get_model_schema_status(dataset, selected_target, model_bundle)
        trained_model = schema_status["trained_model"]
        trained_target = schema_status["trained_target"]
        prediction_features = schema_status["prediction_features"] if not schema_status["prediction_features"].empty else dataset_features
        if isinstance(model_bundle, dict):
            saved_signature = model_bundle.get("dataset_signature")
        dataset_is_trained = saved_signature == active_signature and schema_status["schema_matches"]

    if dataset_is_trained and training_results_df is not None:
        st.markdown("## Training Results")
        render_model_results(training_results_df)
        render_evaluation_panel(evaluation_artifacts or {})
        render_feature_importance_panel(evaluation_artifacts or {})
        render_explainability_panel(evaluation_artifacts or {})


if __name__ == "__main__":
    main()
