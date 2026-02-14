# Verifily CLI v1

One-command workflows for data transformation and model training.

## Install

```bash
cd verifily_cli_v1
pip install typer rich pyyaml
# or: pip install -r requirements.txt
```

## Usage

```bash
python -m verifily_cli_v1 --help
```

## Commands

| Command       | Description                                      |
|---------------|--------------------------------------------------|
| `init`        | Interactive wizard to generate config YAML        |
| `doctor`      | Check environment health (Python, CUDA, deps)     |
| `transform`   | Validate, normalize, and package raw data         |
| `train`       | Launch training via verifily_train                |
| `eval`        | Evaluate a run and display metrics                |
| `compare`     | Compare metrics across multiple runs              |
| `reproduce`   | Verify run integrity via hash chains              |

## Example Flows

### 1. Transform → Train → Eval

```bash
# Generate a transform config
python -m verifily_cli_v1 init

# Transform raw data into a training-ready dataset
python -m verifily_cli_v1 transform \
  --in data/raw_input.jsonl \
  --out datasets/my_dataset_v1

# Preview the training plan (dry run)
python -m verifily_cli_v1 train --config train.yaml --plan

# Launch training
python -m verifily_cli_v1 train --config train.yaml --run-dir runs/exp_01

# Evaluate the run
python -m verifily_cli_v1 eval --run runs/exp_01

# Evaluate with slice breakdown
python -m verifily_cli_v1 eval --run runs/exp_01 --verbose
```

### 2. Compare Runs

```bash
# Compare three runs side by side
python -m verifily_cli_v1 compare \
  --runs runs/model_a_human,runs/model_b_contaminated,runs/model_c_synthetic \
  --metric f1

# Compare with config diff
python -m verifily_cli_v1 compare \
  --runs runs/model_a_human,runs/model_c_synthetic \
  --metric f1 --verbose
```

### 3. Doctor + Reproduce

```bash
# Check environment health
python -m verifily_cli_v1 doctor

# Validate a specific config file
python -m verifily_cli_v1 doctor --config train.yaml

# Verify run reproducibility
python -m verifily_cli_v1 reproduce --run runs/model_c_synthetic

# Verify with hash details
python -m verifily_cli_v1 reproduce --run runs/model_c_synthetic --verbose
```

## Design Principles

- **Clean output** — no stack traces unless `--verbose`
- **Next-step guidance** — every command prints what to do next
- **Artifacts** — every operation writes to a run folder
- **Dry-run support** — `--plan` flag for transform and train
- **Reproducibility** — SHA-256 hash chains for every artifact
