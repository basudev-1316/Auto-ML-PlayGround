setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install -r requirements.txt

run:
	.venv/bin/streamlit run app/streamlit_app.py

api:
	.venv/bin/uvicorn api.main:app --reload

test:
	MPLBACKEND=Agg .venv/bin/python -m pytest tests

benchmark:
	MPLBACKEND=Agg .venv/bin/python -c "from src.benchmarking import run_all_benchmarks; run_all_benchmarks()"
