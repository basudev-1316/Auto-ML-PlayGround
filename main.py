"""Command-line entrypoint helpers for AutoML Playground."""


def main() -> None:
    """Print a quick status message for the production AutoML project."""
    print("AutoML Playground is ready.")
    print("Run the UI with: streamlit run app/streamlit_app.py")
    print("Run the API with: uvicorn api.main:app --reload")
    print("Run tests with: pytest tests")


if __name__ == "__main__":
    main()
