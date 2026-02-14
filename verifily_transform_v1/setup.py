"""Package setup for verifily-transform."""

from setuptools import setup

setup(
    name="verifily-transform",
    version="1.0.0",
    description="Raw data to training-ready datasets in one command",
    packages=["verifily_transform"],
    python_requires=">=3.9",
    install_requires=[
        "click>=8.0.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "llm": ["openai>=1.0.0"],
        "fuzzy": ["datasketch>=1.5.0"],
        "all": ["openai>=1.0.0", "datasketch>=1.5.0"],
    },
    entry_points={
        "console_scripts": [
            "verifily-transform=verifily_transform.cli:main",
        ],
    },
)
