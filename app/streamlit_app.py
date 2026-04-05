"""Professional Streamlit UI for the AutoML Playground application."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    build_model_comparison_dataframe,
    ensure_reports_dir,
    save_confusion_matrix_report,
    save_model_comparison_report,
    save_residual_plot_report,
)
from src.explainability import explain_best_model
from src.model_selection import select_best_model
from src.model_training import train_all_models
from src.problem_detection import detect_problem_type
from src.utils import save_json_report


REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"


def inject_custom_styles() -> None:
    """Apply a dark, animated visual theme to the Streamlit app."""
    st.markdown(
        """
        <style>
        :root {
            --bg-1: #07111f;
            --bg-2: #0d1b2f;
            --panel: rgba(9, 16, 30, 0.78);
            --panel-strong: rgba(8, 13, 24, 0.92);
            --border: rgba(255, 255, 255, 0.08);
            --text: #ecf3ff;
            --muted: #9fb0ca;
            --accent: #42d392;
            --accent-2: #4da3ff;
            --accent-3: #ff8a4c;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(66, 211, 146, 0.14), transparent 26%),
                radial-gradient(circle at top right, rgba(77, 163, 255, 0.16), transparent 22%),
                radial-gradient(circle at bottom center, rgba(255, 138, 76, 0.09), transparent 26%),
                linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
            color: var(--text);
        }

        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }

        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 0.75rem 1rem;
            box-shadow: 0 14px 40px rgba(0, 0, 0, 0.18);
        }

        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            color: var(--text);
        }

        div[data-testid="stFileUploader"],
        div[data-baseweb="select"] > div,
        div[data-testid="stDataFrame"],
        div[data-testid="stExpander"] {
            border-radius: 18px;
        }

        .hero-shell {
            position: relative;
            padding: 1.7rem 1.8rem;
            border-radius: 28px;
            background:
                linear-gradient(145deg, rgba(12, 21, 38, 0.96), rgba(8, 15, 28, 0.92));
            border: 1px solid var(--border);
            box-shadow: 0 24px 90px rgba(0, 0, 0, 0.24);
            overflow: hidden;
            margin-bottom: 1.25rem;
        }

        .hero-shell::before {
            content: "";
            position: absolute;
            width: 220px;
            height: 220px;
            right: -60px;
            top: -90px;
            background: radial-gradient(circle, rgba(77, 163, 255, 0.35), transparent 68%);
            animation: pulseGlow 7s ease-in-out infinite;
        }

        .hero-shell::after {
            content: "";
            position: absolute;
            width: 180px;
            height: 180px;
            left: -40px;
            bottom: -70px;
            background: radial-gradient(circle, rgba(66, 211, 146, 0.28), transparent 72%);
            animation: pulseGlow 6s ease-in-out infinite reverse;
        }

        .hero-tag {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            background: rgba(66, 211, 146, 0.14);
            border: 1px solid rgba(66, 211, 146, 0.2);
            color: #d9ffee;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.76rem;
            font-weight: 700;
            animation: floatY 3.8s ease-in-out infinite;
        }

        .hero-title {
            margin: 0.95rem 0 0 0;
            font-size: clamp(2.7rem, 5vw, 4.7rem);
            line-height: 0.96;
            color: var(--text);
            font-weight: 900;
            letter-spacing: -0.05em;
        }

        .hero-title span {
            background: linear-gradient(90deg, var(--accent), var(--accent-2), var(--accent-3));
            background-size: 200% 200%;
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            animation: shiftGradient 8s ease infinite;
        }

        .hero-subtitle {
            margin-top: 0.7rem;
            color: var(--muted);
            font-size: 1rem;
            max-width: 48rem;
        }

        .hero-line {
            margin-top: 1rem;
            width: 160px;
            height: 4px;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent), var(--accent-2), var(--accent-3));
            background-size: 200% 200%;
            animation: shiftGradient 7s ease infinite;
        }

        .section-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 1.15rem 1.2rem;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.18);
            margin-bottom: 1rem;
            backdrop-filter: blur(16px);
        }

        .section-title {
            margin: 0 0 0.2rem 0;
            color: var(--text);
            font-size: 1.18rem;
            font-weight: 700;
        }

        .section-subtitle {
            margin: 0;
            color: var(--muted);
            font-size: 0.96rem;
        }

        div[data-testid="stButton"] > button {
            min-height: 3.1rem;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.12);
            background: linear-gradient(90deg, rgba(66, 211, 146, 0.24), rgba(77, 163, 255, 0.26));
            color: var(--text);
            font-weight: 800;
            box-shadow: 0 12px 30px rgba(0,0,0,0.16);
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        }

        div[data-testid="stButton"] > button:hover {
            transform: translateY(-1px);
            border-color: rgba(255,255,255,0.24);
            box-shadow: 0 16px 36px rgba(0,0,0,0.24);
        }

        @keyframes shiftGradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes floatY {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-3px); }
            100% { transform: translateY(0px); }
        }

        @keyframes pulseGlow {
            0% { transform: scale(1); opacity: 0.8; }
            50% { transform: scale(1.12); opacity: 1; }
            100% { transform: scale(1); opacity: 0.8; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_selection_ready_results(
    raw_results: dict[str, dict[str, float]],
    trained_models: dict[str, object],
    problem_type: str,
) -> dict[str, dict[str, object]]:
    """Enrich raw training results with metric labels and fitted model objects."""
    metric_name = "Accuracy" if problem_type == "classification" else "RMSE"
    enriched_results: dict[str, dict[str, object]] = {}

    for model_name, metrics in raw_results.items():
        enriched_results[model_name] = {
            **metrics,
            "metric_name": metric_name,
            "model_object": trained_models.get(model_name),
        }

    return enriched_results


def run_training_workflow(dataset: pd.DataFrame, target_column: str) -> dict[str, object]:
    """Execute the full AutoML training workflow for the active dataset."""
    detection = detect_problem_type(dataset, target_column)
    features = dataset.drop(columns=[target_column])
    target = dataset[target_column]

    raw_results, trained_models = train_all_models(features, target, detection.problem_type)
    selection_results = build_selection_ready_results(raw_results, trained_models, detection.problem_type)
    best_model_name, best_model_object, reasoning = select_best_model(selection_results)

    reports_dir = ensure_reports_dir(REPORTS_DIR)
    comparison_csv_path = save_model_comparison_report(
        selection_results,
        reports_dir,
        best_model_name=best_model_name,
    )

    comparison_df = build_model_comparison_dataframe(
        selection_results,
        best_model_name=best_model_name,
    )

    predictions = best_model_object.predict(features)
    if detection.problem_type == "classification":
        task_plot_path = save_confusion_matrix_report(target, predictions, reports_dir)
    else:
        task_plot_path = save_residual_plot_report(target, predictions, reports_dir)

    explanation = explain_best_model(best_model_object, features, reports_dir)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "model": best_model_object,
        "model_name": best_model_name,
        "problem_type": detection.problem_type,
        "target_column": target_column,
        "feature_columns": features.columns.tolist(),
        "score": selection_results[best_model_name]["mean_score"],
        "metric_name": selection_results[best_model_name]["metric_name"],
        "reasoning": reasoning,
    }
    joblib.dump(model_bundle, MODEL_PATH)

    save_json_report(
        {
            "problem_type": detection.problem_type,
            "target_column": target_column,
            "best_model_name": best_model_name,
            "best_score": selection_results[best_model_name]["mean_score"],
            "metric_name": selection_results[best_model_name]["metric_name"],
            "reasoning": reasoning,
            "reports": {
                "comparison_csv": str(comparison_csv_path),
                "task_plot": str(task_plot_path),
                "feature_importance": str(explanation.feature_importance_path),
                "shap_summary": str(explanation.summary_plot_path),
            },
            "models": list(selection_results.keys()),
        },
        reports_dir / "deployment_summary.json",
    )

    return {
        "problem_type": detection.problem_type,
        "feature_types": detection.feature_types,
        "results": selection_results,
        "comparison_df": comparison_df,
        "comparison_csv_path": comparison_csv_path,
        "best_model_name": best_model_name,
        "best_model": best_model_object,
        "best_score": selection_results[best_model_name]["mean_score"],
        "metric_name": selection_results[best_model_name]["metric_name"],
        "reasoning": reasoning,
        "task_plot_path": task_plot_path,
        "explanation": explanation,
        "saved_model_path": MODEL_PATH,
    }


def render_summary_cards(training_output: dict[str, object]) -> None:
    """Render key summary metrics for the completed training run."""
    col_1, col_2, col_3 = st.columns(3)
    with col_1:
        st.metric("Problem Type", str(training_output["problem_type"]).title())
    with col_2:
        st.metric("Best Model", str(training_output["best_model_name"]))
    with col_3:
        st.metric(
            str(training_output["metric_name"]),
            f"{float(training_output['best_score']):.4f}",
        )


def render_model_comparison_chart(comparison_df: pd.DataFrame) -> None:
    """Render the model comparison chart in the Streamlit app."""
    metric_name = str(comparison_df["metric_name"].iloc[0])
    lower_is_better = "RMSE" in metric_name
    plot_df = comparison_df.sort_values("mean_score", ascending=lower_is_better)

    fig = px.bar(
        plot_df,
        x="mean_score",
        y="model_name",
        color="highlight",
        orientation="h",
        text="mean_score",
        color_discrete_map={
            "Best Model": "#42d392",
            "Other Models": "#4da3ff",
        },
        title=f"Model Comparison ({metric_name})",
        hover_data=["rank", "std_score", "train_time"],
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(
        height=480,
        xaxis_title=metric_name,
        yaxis_title="Model",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ecf3ff"},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_shap_section(training_output: dict[str, object]) -> None:
    """Render SHAP outputs for the selected best model."""
    explanation = training_output["explanation"]
    st.subheader("Model Explainability")
    st.caption("SHAP-based explanation for the selected best model.")

    plot_col_1, plot_col_2 = st.columns(2)
    with plot_col_1:
        st.image(
            str(explanation.feature_importance_path),
            caption="Feature Importance Plot",
            use_container_width=True,
        )
    with plot_col_2:
        st.image(
            str(explanation.summary_plot_path),
            caption="SHAP Summary Plot",
            use_container_width=True,
        )
    st.dataframe(explanation.importance_frame.head(20), use_container_width=True)


def render_hero() -> None:
    """Render the animated dark hero section."""
    st.markdown(
        """
        <section class="hero-shell">
            <div class="hero-tag">Explainable AutoML</div>
            <h1 class="hero-title">Auto<span>ML Playground</span></h1>
            <p class="hero-subtitle">
                Explore your dataset first, understand its quality and structure, and then train an
                explainable AutoML system that benchmarks models, selects the best candidate, and
                saves production-ready artifacts.
            </p>
            <div class="hero-line"></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_summary(dataset: pd.DataFrame, target_column: str) -> None:
    """Render a compact overview of the uploaded dataset."""
    numeric_columns = dataset.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in dataset.columns if column not in numeric_columns]

    col_1, col_2, col_3, col_4 = st.columns(4)
    with col_1:
        st.metric("Rows", f"{len(dataset):,}")
    with col_2:
        st.metric("Columns", len(dataset.columns))
    with col_3:
        st.metric("Numeric Features", len([column for column in numeric_columns if column != target_column]))
    with col_4:
        st.metric("Categorical Features", len([column for column in categorical_columns if column != target_column]))


def render_complete_eda(dataset: pd.DataFrame, target_column: str) -> None:
    """Render a full pre-training EDA experience for the uploaded dataset."""
    numeric_columns = dataset.select_dtypes(include=["number"]).columns.tolist()
    feature_numeric_columns = [column for column in numeric_columns if column != target_column]
    categorical_columns = [column for column in dataset.columns if column not in numeric_columns]
    feature_categorical_columns = [column for column in categorical_columns if column != target_column]

    st.markdown(
        """
        <section class="section-card">
            <h3 class="section-title">Dataset Explorer</h3>
            <p class="section-subtitle">Inspect the dataset before training the model.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    render_dataset_summary(dataset, target_column)

    overview_tab, quality_tab, numeric_tab, categorical_tab = st.tabs(
        ["Preview", "Quality", "Numeric EDA", "Categorical EDA"]
    )

    with overview_tab:
        st.dataframe(dataset.head(20), use_container_width=True)
        st.dataframe(dataset.describe(include="all").transpose().fillna(""), use_container_width=True)

    with quality_tab:
        missing_counts = dataset.isna().sum().sort_values(ascending=False)
        missing_df = missing_counts[missing_counts > 0].reset_index()
        missing_df.columns = ["column", "missing_values"]

        quality_col_1, quality_col_2 = st.columns(2)
        with quality_col_1:
            st.metric("Missing Cells", int(dataset.isna().sum().sum()))
        with quality_col_2:
            st.metric("Duplicate Rows", int(dataset.duplicated().sum()))

        if not missing_df.empty:
            missing_fig = px.bar(
                missing_df,
                x="column",
                y="missing_values",
                title="Missing Values by Column",
                color="missing_values",
                color_continuous_scale="Tealgrn",
            )
            missing_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#ecf3ff"},
            )
            st.plotly_chart(missing_fig, use_container_width=True)
        else:
            st.success("No missing values found in the dataset.")

        if target_column in numeric_columns:
            target_fig = px.histogram(
                dataset,
                x=target_column,
                nbins=30,
                title=f"Target Distribution: {target_column}",
                color_discrete_sequence=["#42d392"],
            )
        else:
            target_counts = dataset[target_column].astype(str).value_counts(dropna=False).reset_index()
            target_counts.columns = ["target_value", "count"]
            target_fig = px.bar(
                target_counts,
                x="target_value",
                y="count",
                title=f"Target Class Balance: {target_column}",
                color="count",
                color_continuous_scale="Tealgrn",
            )
        target_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#ecf3ff"},
        )
        st.plotly_chart(target_fig, use_container_width=True)

    with numeric_tab:
        if feature_numeric_columns:
            selected_numeric = feature_numeric_columns[: min(6, len(feature_numeric_columns))]
            numeric_long_df = dataset[selected_numeric].melt(var_name="feature", value_name="value").dropna()
            if not numeric_long_df.empty:
                distribution_fig = px.histogram(
                    numeric_long_df,
                    x="value",
                    facet_col="feature",
                    facet_col_wrap=3,
                    nbins=25,
                    title="Numeric Feature Distributions",
                    color_discrete_sequence=["#4da3ff"],
                )
                distribution_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#ecf3ff"},
                )
                st.plotly_chart(distribution_fig, use_container_width=True)

            if len(feature_numeric_columns) >= 2:
                correlation_columns = feature_numeric_columns + ([target_column] if target_column in numeric_columns else [])
                correlation_df = dataset[correlation_columns].corr(numeric_only=True)
                corr_fig = px.imshow(
                    correlation_df,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="Tealgrn",
                    title="Correlation Heatmap",
                )
                corr_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#ecf3ff"},
                )
                st.plotly_chart(corr_fig, use_container_width=True)

                if target_column in correlation_df.columns:
                    target_corr_df = (
                        correlation_df[target_column]
                        .drop(labels=[target_column], errors="ignore")
                        .abs()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index()
                    )
                    target_corr_df.columns = ["feature", "absolute_correlation"]
                    target_corr_fig = px.bar(
                        target_corr_df,
                        x="feature",
                        y="absolute_correlation",
                        title=f"Top Numeric Relationships with {target_column}",
                        color="absolute_correlation",
                        color_continuous_scale="Tealgrn",
                    )
                    target_corr_fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": "#ecf3ff"},
                    )
                    st.plotly_chart(target_corr_fig, use_container_width=True)
        else:
            st.info("No numeric feature columns found for numeric EDA.")

    with categorical_tab:
        if feature_categorical_columns:
            selected_categorical = feature_categorical_columns[: min(4, len(feature_categorical_columns))]
            for column in selected_categorical:
                counts_df = dataset[column].astype(str).value_counts(dropna=False).head(15).reset_index()
                counts_df.columns = [column, "count"]
                cat_fig = px.bar(
                    counts_df,
                    x=column,
                    y="count",
                    title=f"Top Categories: {column}",
                    color="count",
                    color_continuous_scale="Tealgrn",
                )
                cat_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#ecf3ff"},
                )
                st.plotly_chart(cat_fig, use_container_width=True)
        else:
            st.info("No categorical feature columns found for categorical EDA.")


def main() -> None:
    """Render the full production Streamlit application."""
    st.set_page_config(page_title="AutoML Playground", page_icon="ML", layout="wide")
    inject_custom_styles()
    render_hero()

    st.markdown(
        """
        <section class="section-card">
            <h3 class="section-title">Training Setup</h3>
            <p class="section-subtitle">Upload a dataset, inspect it, and train the AutoML system when you are ready.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV file to begin.")
        return

    try:
        dataset = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Failed to read the uploaded CSV: {exc}")
        return

    if dataset.empty:
        st.error("The uploaded dataset is empty.")
        return

    target_column = st.selectbox("Select target column", options=dataset.columns.tolist())

    try:
        detection_preview = detect_problem_type(dataset, target_column)
        preview_col_1, preview_col_2, preview_col_3 = st.columns(3)
        with preview_col_1:
            st.metric("Detected Problem Type", detection_preview.problem_type.title())
        with preview_col_2:
            st.metric("Numeric Features", len(detection_preview.feature_types.numeric_columns))
        with preview_col_3:
            st.metric("Categorical Features", len(detection_preview.feature_types.categorical_columns))
    except Exception as exc:
        st.warning(f"Problem detection warning: {exc}")

    render_complete_eda(dataset, target_column)

    if st.button("Train Model", use_container_width=True):
        try:
            with st.spinner("Training models, selecting the winner, generating reports, and preparing SHAP explainability..."):
                training_output = run_training_workflow(dataset, target_column)
            st.success("Training completed successfully.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            return

        render_summary_cards(training_output)

        result_tab_1, result_tab_2, result_tab_3 = st.tabs(
            ["Selection", "Comparison", "Explainability"]
        )

        with result_tab_1:
            st.subheader("Selection Reasoning")
            st.write(training_output["reasoning"])
            st.subheader("Task-Specific Evaluation Plot")
            st.image(str(training_output["task_plot_path"]), use_container_width=True)

        with result_tab_2:
            st.subheader("Model Comparison")
            render_model_comparison_chart(training_output["comparison_df"])
            st.dataframe(training_output["comparison_df"], use_container_width=True)

        with result_tab_3:
            render_shap_section(training_output)

        st.subheader("Saved Outputs")
        output_col_1, output_col_2 = st.columns(2)
        with output_col_1:
            st.success(f"Reports saved in `{REPORTS_DIR}`")
        with output_col_2:
            st.success(f"Trained model saved in `{training_output['saved_model_path']}`")


if __name__ == "__main__":
    main()
