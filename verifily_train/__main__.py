"""Allow running as `python -m verifily_train`."""
import os

_orig_warnings = os.environ.get("PYTHONWARNINGS")
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.disable(logging.WARNING)

from verifily_train.cli import main  # noqa: E402

logging.disable(logging.NOTSET)
warnings.resetwarnings()
if _orig_warnings is not None:
    os.environ["PYTHONWARNINGS"] = _orig_warnings
else:
    os.environ.pop("PYTHONWARNINGS", None)

main()
