import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "app" / "streamlit_ui.py"
SPEC = importlib.util.spec_from_file_location("streamlit_ui", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
main = MODULE.main


if __name__ == "__main__":
    main()
