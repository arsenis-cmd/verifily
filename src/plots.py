"""Generate plots and visualizations for experiment results."""
import os
import logging
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import jsonlines

from src.utils import ensure_dir

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


class ResultsVisualizer:
    """Generate plots and visualizations from evaluation results."""

    def __init__(self, config: Dict):
        self.config = config
        self.results_dir = config.get("results_dir", "results")
        self.plots_dir = os.path.join(self.results_dir, "plots")
        ensure_dir(self.plots_dir)

    def load_results(self, metrics_file: str = None) -> pd.DataFrame:
        """Load evaluation results from JSONL file."""
        if metrics_file is None:
            metrics_file = os.path.join(self.results_dir, "metrics.jsonl")

        logger.info(f"Loading results from {metrics_file}")

        results = []
        with jsonlines.open(metrics_file) as reader:
            for obj in reader:
                row = {"model": obj["model_id"]}
                row.update(obj["metrics"])
                results.append(row)

        df = pd.DataFrame(results)
        logger.info(f"Loaded {len(df)} model results")

        return df

    def create_comparison_barplot(self, df: pd.DataFrame, metric: str = "f1"):
        """Create bar plot comparing models on a single metric."""
        logger.info(f"Creating bar plot for metric: {metric}")

        plt.figure(figsize=(10, 6))

        # Create bar plot
        models = df["model"].tolist()
        values = df[metric].tolist()

        # Color mapping
        colors = []
        for model in models:
            if "human" in model.lower():
                colors.append("#2ecc71")  # Green for human baseline
            elif "contaminated" in model.lower():
                colors.append("#e74c3c")  # Red for contaminated
            elif "synthetic" in model.lower():
                colors.append("#3498db")  # Blue for synthetic
            else:
                colors.append("#95a5a6")  # Gray for others

        bars = plt.bar(range(len(models)), values, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{value:.3f}',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )

        # Customize
        plt.xlabel("Model", fontsize=14, fontweight='bold')
        plt.ylabel(metric.upper().replace("_", " "), fontsize=14, fontweight='bold')
        plt.title(f"Model Comparison: {metric.upper()}", fontsize=16, fontweight='bold', pad=20)
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        plt.ylim(0, max(values) * 1.15)
        plt.tight_layout()

        # Save
        output_path = os.path.join(self.plots_dir, f"comparison_{metric}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved plot to {output_path}")
        return output_path

    def create_multi_metric_comparison(self, df: pd.DataFrame, metrics: List[str] = None):
        """Create grouped bar plot comparing models across multiple metrics."""
        if metrics is None:
            # Use all numeric columns except 'model'
            metrics = [col for col in df.columns if col != "model" and df[col].dtype in ['float64', 'int64']]

        logger.info(f"Creating multi-metric comparison for: {metrics}")

        # Prepare data for plotting
        models = df["model"].tolist()
        x = range(len(models))
        width = 0.8 / len(metrics)

        plt.figure(figsize=(14, 7))

        # Create grouped bars
        for i, metric in enumerate(metrics):
            offset = (i - len(metrics) / 2) * width + width / 2
            values = df[metric].tolist()
            plt.bar(
                [pos + offset for pos in x],
                values,
                width,
                label=metric.upper().replace("_", " "),
                alpha=0.8
            )

        # Customize
        plt.xlabel("Model", fontsize=14, fontweight='bold')
        plt.ylabel("Score", fontsize=14, fontweight='bold')
        plt.title("Model Comparison: Multiple Metrics", fontsize=16, fontweight='bold', pad=20)
        plt.xticks(x, models, rotation=45, ha='right')
        plt.legend(loc='upper left', frameon=True, shadow=True)
        plt.ylim(0, 1.0)
        plt.tight_layout()

        # Save
        output_path = os.path.join(self.plots_dir, "comparison_multi_metric.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved plot to {output_path}")
        return output_path

    def create_heatmap(self, df: pd.DataFrame):
        """Create heatmap of all metrics across models."""
        logger.info("Creating heatmap")

        # Prepare data
        numeric_cols = [col for col in df.columns if col != "model" and df[col].dtype in ['float64', 'int64']]
        data = df[numeric_cols].T  # Transpose so metrics are rows

        plt.figure(figsize=(10, max(6, len(numeric_cols) * 0.8)))

        # Create heatmap
        sns.heatmap(
            data,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            cbar_kws={'label': 'Score'},
            xticklabels=df["model"].tolist(),
            yticklabels=[col.upper().replace("_", " ") for col in numeric_cols],
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor='gray'
        )

        plt.title("Model Performance Heatmap", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Model", fontsize=14, fontweight='bold')
        plt.ylabel("Metric", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save
        output_path = os.path.join(self.plots_dir, "heatmap.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved plot to {output_path}")
        return output_path

    def create_performance_gap_plot(self, df: pd.DataFrame, metric: str = "f1"):
        """Create plot showing performance gap between models."""
        logger.info(f"Creating performance gap plot for {metric}")

        # Assume model_a_human is the baseline
        baseline_row = df[df["model"].str.contains("human", case=False)]
        if baseline_row.empty:
            logger.warning("No human baseline found, using first model")
            baseline_value = df[metric].iloc[0]
        else:
            baseline_value = baseline_row[metric].iloc[0]

        # Calculate gaps
        models = df["model"].tolist()
        values = df[metric].tolist()
        gaps = [v - baseline_value for v in values]

        plt.figure(figsize=(10, 6))

        # Create bar plot with positive/negative colors
        colors = ['#2ecc71' if g >= 0 else '#e74c3c' for g in gaps]
        bars = plt.bar(range(len(models)), gaps, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels
        for i, (bar, gap, value) in enumerate(zip(bars, gaps, values)):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.01 if gap >= 0 else -0.01),
                f'{gap:+.3f}\n({value:.3f})',
                ha='center',
                va='bottom' if gap >= 0 else 'top',
                fontsize=10,
                fontweight='bold'
            )

        # Add baseline line
        plt.axhline(y=0, color='black', linestyle='--', linewidth=2, label=f'Baseline ({baseline_value:.3f})')

        # Customize
        plt.xlabel("Model", fontsize=14, fontweight='bold')
        plt.ylabel(f"{metric.upper()} Gap from Baseline", fontsize=14, fontweight='bold')
        plt.title(f"Performance Gap Analysis: {metric.upper()}", fontsize=16, fontweight='bold', pad=20)
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()

        # Save
        output_path = os.path.join(self.plots_dir, f"gap_analysis_{metric}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved plot to {output_path}")
        return output_path

    def generate_all_plots(self, metrics_file: str = None):
        """Generate all plots from evaluation results."""
        logger.info("="*50)
        logger.info("GENERATING PLOTS")
        logger.info("="*50)

        # Load results
        df = self.load_results(metrics_file)

        # Generate plots
        plots = []

        # Main comparison plot (F1)
        if "f1" in df.columns:
            plots.append(self.create_comparison_barplot(df, "f1"))

        # Exact match comparison
        if "exact_match" in df.columns:
            plots.append(self.create_comparison_barplot(df, "exact_match"))

        # Multi-metric comparison
        plots.append(self.create_multi_metric_comparison(df))

        # Heatmap
        plots.append(self.create_heatmap(df))

        # Gap analysis
        if "f1" in df.columns:
            plots.append(self.create_performance_gap_plot(df, "f1"))

        logger.info(f"Generated {len(plots)} plots")
        logger.info("="*50)

        return plots


def main():
    """CLI entry point for generating plots."""
    import argparse
    from src.utils import load_config, setup_logging

    parser = argparse.ArgumentParser(description="Generate plots from evaluation results")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default=None,
        help="Path to metrics JSONL file (defaults to results/metrics.jsonl)"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup logging
    setup_logging(
        log_file=config.get("logging", {}).get("log_file"),
        level=config.get("logging", {}).get("level", "INFO")
    )

    # Generate plots
    visualizer = ResultsVisualizer(config)
    plots = visualizer.generate_all_plots(args.metrics_file)

    print("\nPlots generated successfully!")
    print(f"Saved to: {visualizer.plots_dir}")
    for plot in plots:
        print(f"  - {plot}")


if __name__ == "__main__":
    main()
