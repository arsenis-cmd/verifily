# Verifily Train v1 -- Specification

**Status**: Draft (implementation-ready)
**Version**: 1.0.0-draft
**Date**: 2026-02-08

## What is this?

This directory contains the complete product and engineering specification for **Verifily Train v1** -- a dataset-aware training adapter that makes training on Verifily dataset versions a one-command operation.

Verifily Train is **not** a new training framework or optimizer. It wraps proven stacks (HuggingFace Transformers, PEFT/LoRA, Accelerate) and adds dataset versioning, reproducibility, and evaluation as first-class concerns.

## Documents

| File | Purpose |
|------|---------|
| [spec.md](spec.md) | Core product specification: goals, non-goals, primitives, config schemas, MVP plan |
| [cli.md](cli.md) | CLI design: all commands, flags, example invocations |
| [api.md](api.md) | Python API: core classes, methods, integration points |
| [run_artifacts.md](run_artifacts.md) | Run artifact layout: exact folder tree and file purposes |
| [eval.md](eval.md) | Evaluation framework: metrics, slicing, dataset attribution |
| [security_privacy.md](security_privacy.md) | Security and privacy posture: managed vs self-host, data boundaries |
| [roadmap.md](roadmap.md) | Implementation plan, 2-week MVP scope, pricing hooks, future versions |

## Quick Context

Verifily Train sits between a customer's dataset (managed by the Verifily platform) and their fine-tuned model. The core value proposition:

1. Customer versions their dataset in Verifily (human, synthetic, mixed).
2. `verifily train` pulls the dataset version and runs fine-tuning with sensible defaults.
3. `verifily eval` evaluates the resulting model, slicing metrics by dataset tags.
4. `verifily compare` shows how different dataset versions affect model quality.

The insight this enables: **which data matters, and which data hurts**.
