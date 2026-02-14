"""Allow running as `python -m verifily_transform`."""
import warnings
warnings.filterwarnings("ignore")

from verifily_transform.cli import main

main()
