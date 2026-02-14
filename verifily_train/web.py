"""Verifily Train Web UI — lightweight Gradio dashboard for browsing runs."""

import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _discover_runs(base_dir: str) -> List[dict]:
    """Scan a directory for run artifacts."""
    runs = []
    base = Path(base_dir)
    if not base.exists():
        return runs
    for d in sorted(base.iterdir(), reverse=True):
        meta = d / "run_meta.json"
        if meta.exists():
            with open(meta) as f:
                runs.append(json.load(f))
    return runs


def _load_eval(artifact_path: str) -> Optional[dict]:
    p = Path(artifact_path) / "eval" / "eval_results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _load_hard_examples(artifact_path: str) -> List[dict]:
    p = Path(artifact_path) / "eval" / "hard_examples.jsonl"
    if not p.exists():
        return []
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def launch(runs_dir: str = "runs/", port: int = 7860, share: bool = False):
    """Launch the Gradio web UI."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError(
            "Gradio is required for the web UI. Install it: pip install gradio"
        )

    def get_run_list():
        runs = _discover_runs(runs_dir)
        if not runs:
            return "No runs found."
        lines = []
        for r in runs:
            status = r.get("status", "?")
            rid = r.get("run_id", "unknown")
            task = r.get("task", "?")
            model = r.get("base_model", "?")
            loss = r.get("metrics", {}).get("train_loss", "N/A")
            dur = r.get("duration_seconds", 0)
            lines.append(f"**{rid}** | {status} | {task} | {model} | loss={loss} | {dur}s")
        return "\n\n".join(lines)

    def get_run_detail(run_id: str):
        runs = _discover_runs(runs_dir)
        for r in runs:
            if r.get("run_id") == run_id:
                ev = _load_eval(r.get("artifact_path", ""))
                output = f"## Run: {run_id}\n\n"
                output += f"**Status:** {r.get('status')}\n\n"
                output += f"**Task:** {r.get('task')}\n\n"
                output += f"**Model:** {r.get('base_model')}\n\n"
                output += f"**Device:** {r.get('device')}\n\n"
                output += f"**Duration:** {r.get('duration_seconds')}s\n\n"
                output += f"**Seed:** {r.get('seed')}\n\n"
                output += f"### Training Metrics\n"
                for k, v in r.get("metrics", {}).items():
                    output += f"- {k}: {v}\n"
                if ev:
                    output += f"\n### Evaluation Metrics\n"
                    for k, v in ev.get("overall", {}).items():
                        if isinstance(v, float):
                            output += f"- {k}: {v:.4f}\n"
                        else:
                            output += f"- {k}: {v}\n"
                    if ev.get("slices"):
                        output += f"\n### Slices\n"
                        for tag_key, tag_vals in ev["slices"].items():
                            output += f"\n**{tag_key}:**\n"
                            for tv, tv_m in tag_vals.items():
                                output += f"- {tv}: {tv_m}\n"
                output += f"\n### Hashes\n"
                output += f"- Config: `{r.get('config_hash', 'N/A')}`\n"
                output += f"- Data: `{r.get('data_hash', 'N/A')}`\n"
                output += f"- Repro: `{r.get('reproducibility_hash', 'N/A')}`\n"
                return output
        return f"Run '{run_id}' not found."

    def get_hard_examples(run_id: str):
        runs = _discover_runs(runs_dir)
        for r in runs:
            if r.get("run_id") == run_id:
                examples = _load_hard_examples(r.get("artifact_path", ""))
                if not examples:
                    return "No hard examples found."
                lines = []
                for ex in examples[:20]:
                    lines.append(
                        f"**#{ex['rank']}** (F1={ex.get('f1', 0):.2f})\n"
                        f"- **Input:** {ex.get('input', '')[:200]}\n"
                        f"- **Predicted:** {ex.get('prediction', '')}\n"
                        f"- **Expected:** {ex.get('reference', '')}\n"
                    )
                return "\n---\n".join(lines)
        return f"Run '{run_id}' not found."

    def get_compare(run_ids_str: str, metric: str):
        run_ids = [r.strip() for r in run_ids_str.split(",") if r.strip()]
        if len(run_ids) < 2:
            return "Enter at least 2 run IDs separated by commas."
        runs = _discover_runs(runs_dir)
        id_to_path = {r["run_id"]: r["artifact_path"] for r in runs}
        paths = []
        for rid in run_ids:
            if rid not in id_to_path:
                return f"Run '{rid}' not found."
            paths.append(id_to_path[rid])
        try:
            from verifily_train.compare import compare, format_comparison
            result = compare(paths, metric=metric)
            return f"```\n{format_comparison(result)}\n```"
        except Exception as e:
            return f"Error: {e}"

    def run_id_choices():
        runs = _discover_runs(runs_dir)
        return [r.get("run_id", "unknown") for r in runs]

    with gr.Blocks(title="Verifily Train Dashboard") as app:
        gr.Markdown("# Verifily Train Dashboard")

        with gr.Tab("Runs"):
            gr.Markdown("### All Runs")
            run_list_out = gr.Markdown()
            refresh_btn = gr.Button("Refresh")
            refresh_btn.click(fn=get_run_list, outputs=run_list_out)
            app.load(fn=get_run_list, outputs=run_list_out)

        with gr.Tab("Run Detail"):
            run_id_input = gr.Textbox(label="Run ID")
            detail_btn = gr.Button("Load")
            detail_out = gr.Markdown()
            detail_btn.click(fn=get_run_detail, inputs=run_id_input, outputs=detail_out)

        with gr.Tab("Hard Examples"):
            hard_run_input = gr.Textbox(label="Run ID")
            hard_btn = gr.Button("Load Hard Examples")
            hard_out = gr.Markdown()
            hard_btn.click(fn=get_hard_examples, inputs=hard_run_input, outputs=hard_out)

        with gr.Tab("Compare"):
            compare_ids = gr.Textbox(label="Run IDs (comma-separated)")
            compare_metric = gr.Textbox(label="Metric", value="f1")
            compare_btn = gr.Button("Compare")
            compare_out = gr.Markdown()
            compare_btn.click(
                fn=get_compare, inputs=[compare_ids, compare_metric], outputs=compare_out,
            )

    logger.info("Launching Verifily dashboard on port %d", port)
    app.launch(server_port=port, share=share)
