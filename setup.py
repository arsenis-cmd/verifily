"""Package setup for verifily-train."""

from setuptools import setup, find_packages

setup(
    name="verifily-train",
    version="1.1.0",
    description="Dataset-aware fine-tuning in one command",
    packages=["verifily_train"],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "peft>=0.6.0",
        "accelerate>=0.24.0",
        "evaluate>=0.4.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "click>=8.0.0",
    ],
    extras_require={
        "qlora": ["bitsandbytes>=0.42.0"],
        "wandb": ["wandb>=0.16.0"],
        "web": ["gradio>=4.0.0"],
        "rouge": ["rouge-score>=0.1.2"],
        "all": ["bitsandbytes>=0.42.0", "wandb>=0.16.0", "gradio>=4.0.0", "rouge-score>=0.1.2"],
    },
    entry_points={
        "console_scripts": [
            "verifily=verifily_train.cli:main",
        ],
    },
)
