"""Entry point for python -m verifily_cli_v1."""

import sys
import warnings

warnings.filterwarnings("ignore")

from verifily_cli_v1.cli import app  # noqa: E402

if __name__ == "__main__":
    app()
