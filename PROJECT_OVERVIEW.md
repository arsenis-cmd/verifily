# ML Experiment: Human vs AI-Contaminated vs Synthetic Data

## Experiment Goal
Prove that models trained on **synthetic data derived from human sources** outperform models trained on **AI-contaminated data**, demonstrating the value of human-derived synthetic generation.

## Three-Model Comparison

### Model A (Human Baseline)
- **Training data**: 20,000 human-annotated QA pairs from SQuAD dataset
- **Status**: ✅ COMPLETE
- **Performance**:
  - Exact Match: 59.45%
  - F1 Score: 71.39%
- **Location**: `runs/model_a_human/`

### Model B (AI-Contaminated)
- **Training data**: 20,000 QA pairs with same questions as Model A, but AI-generated answers
- **Status**: ✅ COMPLETE
- **Performance**:
  - Exact Match: 58.80%
  - F1 Score: 70.57%
- **Contamination effect**: -0.82 F1 points vs Model A
- **Location**: `runs/model_b_contaminated/`

### Model C (Synthetic from Human Seed)
- **Training data**: 20,000 synthetic QA pairs generated from human seed data using FLAN-T5-XL
- **Status**: ⏳ IN PROGRESS - awaiting proper synthetic data generation
- **Previous attempt**: FAILED (0% EM, 7% F1) due to broken synthetic data
- **Location**: `runs/model_c_synthetic/` (needs to be retrained)

## Expected Outcome
**Model C > Model B** and **Model C ≈ Model A**

This would prove that synthetic data derived from human sources is superior to AI-contaminated data.

---

## Current Status (as of 2026-02-08)

### Completed ✅
1. Data preparation (20k human samples)
2. AI contamination generation (GPT-3.5-turbo)
3. Model A training and evaluation
4. Model B training and evaluation
5. Initial results comparison showing contamination degradation

### In Progress ⏳
1. **FLAN-T5-XL model download** (PID: 53002)
   - Size: ~11GB
   - Status: Downloading from scratch after corrupted cache cleanup
   - ETA: 30-60 minutes

### Pending Tasks 📋
1. Test synthesis with 2 samples to verify quality
2. Generate 20,000 synthetic samples (~6-8 hours)
3. Retrain Model C on proper synthetic data (~40 minutes)
4. Re-evaluate Model C
5. Generate final comparison plots

---

## Technical Architecture

### Base Model
- **FLAN-T5-base** (250M parameters) - used for Models A, B, C training
- Fine-tuned using LoRA (Low-Rank Adaptation)
- Training: ~40 minutes on MPS GPU

### Synthesis Model
- **FLAN-T5-XL** (3B parameters) - for generating synthetic data
- Larger model needed for quality synthetic generation
- Previous attempts with FLAN-T5-base failed (too weak)

### Hardware
- Device: Apple Silicon with MPS (Metal Performance Shaders) GPU
- Fallback: CPU for longer tasks

---

## Key Files and Directories

### Configuration Files
- `configs/base.yaml` - Main training configuration
- `configs/synth_20k.yaml` - Synthetic data generation config (FLAN-T5-XL)

### Data Files
- `data/processed/human_train.jsonl` - 20k human QA pairs
- `data/processed/ai_contaminated_train.jsonl` - 20k contaminated QA pairs
- `data/synthetic/synthetic_train.jsonl` - Synthetic QA pairs (pending proper generation)
- `data/processed/human_val.jsonl` - Validation set
- `data/processed/human_test.jsonl` - Test set (2000 samples)

### Source Code
- `src/prepare_data.py` - Data preprocessing
- `src/contaminate.py` - Generate AI-contaminated answers
- `src/synthesize.py` - Generate synthetic QA pairs (currently being fixed)
- `src/train.py` - Model training with LoRA
- `src/eval.py` - Evaluation on test set

### Results
- `results/metrics_table.csv` - Performance comparison table
- `runs/model_a_human/` - Model A checkpoint and training info
- `runs/model_b_contaminated/` - Model B checkpoint and training info
- `runs/model_c_synthetic/` - Model C (needs retraining)

### Logs
- `test_xl.log` - Current FLAN-T5-XL download/synthesis test
- `training_model_a.log` - Model A training log (completed)
- `training_model_b.log` - Model B training log (completed)

---

## Data Format

All QA pairs follow SQuAD format:
```json
{
  "id": "example_id",
  "question": "What is the capital of France?",
  "context": "Paris is the capital and most populous city of France...",
  "answer": "Paris",
  "source": "human|ai_contaminated|synthetic_from_human"
}
```

---

## Synthesis Approach (Current Strategy)

### Previous Failed Attempts
1. **Attempt 1**: FLAN-T5-base with complex prompts → malformed data (empty contexts, questions as answers)
2. **Attempt 2**: Question paraphrasing with FLAN-T5-base → all samples rejected by quality filters

### Current Approach
- **Model**: FLAN-T5-XL (3B parameters)
- **Strategy**: Paraphrase questions while keeping original context and answers
- **Quality Filters**:
  - Remove exact duplicates
  - N-gram overlap threshold (8-gram, max 40% overlap)
  - Min question length: 8 tokens
  - Min answer length: 1 token
  - Max answer length: 200 tokens

### Generation Parameters
```yaml
temperature: 0.8
top_p: 0.95
max_new_tokens: 256
do_sample: true
```

---

## Important Issues Resolved

### Issue 1: Broken Synthetic Data (First 20k Generation)
- **Problem**: FLAN-T5-base too weak, generated malformed data
- **Impact**: Model C trained on broken data → 0% EM, 7% F1 (catastrophic failure)
- **Solution**: Switch to FLAN-T5-XL (3B params)

### Issue 2: Paraphrases Rejected by Filters
- **Problem**: FLAN-T5-base just copied questions → duplicate filter rejection
- **Solution**: Use stronger FLAN-T5-XL model for better paraphrasing

### Issue 3: Corrupted Model Cache
- **Problem**: Killed download mid-way → stuck at 7.1GB
- **Solution**: Deleted cache, restarting fresh download

---

## Monitoring Commands

### Check FLAN-T5-XL download progress:
```bash
du -sh ~/.cache/huggingface/hub/models--google--flan-t5-xl
```

### Monitor download in real-time:
```bash
while true; do clear; date; du -sh ~/.cache/huggingface/hub/models--google--flan-t5-xl; echo "Target: ~11GB"; sleep 10; done
```

### Check synthesis process:
```bash
tail -f test_xl.log
```

### Check generated samples:
```bash
cat data/synthetic/synthetic_train.jsonl | head -3
```

---

## Next Steps After Download Completes

1. **Verify 2-sample test quality**
   ```bash
   cat data/synthetic/synthetic_train.jsonl
   ```
   Check for:
   - Non-empty contexts
   - Paraphrased questions (different from original)
   - Valid answers from context

2. **If quality is good → Generate full 20k**
   ```bash
   python3 -m src.synthesize --config configs/synth_20k.yaml
   ```
   ETA: 6-8 hours

3. **Retrain Model C**
   ```bash
   rm -rf runs/model_c_synthetic
   python3 -m src.train --config configs/base.yaml \
       --model-id model_c_synthetic \
       --train-data data/synthetic/synthetic_train.jsonl \
       --val-data data/processed/human_val.jsonl
   ```
   ETA: ~40 minutes

4. **Re-evaluate Model C**
   ```bash
   python3 -m src.eval --config configs/base.yaml \
       --models model_c_synthetic:runs/model_c_synthetic \
       --test-data data/processed/human_test.jsonl
   ```
   ETA: ~20-30 minutes

5. **Generate comparison plots**
   - Compare all three models
   - Visualize A vs B vs C performance

---

## Critical Context for New Agent

### User's Core Requirements
- Must use **stronger model** (3B+ params) for synthesis - user explicitly rejected shortcuts
- No copying human data directly - must be **proper synthetic generation**
- Must prove synthetic > contaminated through rigorous experimentation

### Process Currently Running
- **PID 53002**: FLAN-T5-XL download (started fresh after cache corruption)
- Monitor with: `ps aux | grep 53002`

### Known Working Configuration
- Training: FLAN-T5-base + LoRA works well (Models A & B successful)
- Synthesis: FLAN-T5-XL required (base model too weak)
- Evaluation metrics: EM and F1 (ROUGE also available)

### Time Estimates
- FLAN-T5-XL download: 30-60 min (11GB)
- 2-sample test: 1-2 min after download
- 20k synthesis: 6-8 hours
- Model training: ~40 min
- Model evaluation: ~20-30 min

---

## Success Criteria

**Experiment succeeds if:**
- Model C F1 > Model B F1 (synthetic beats contaminated)
- Model C F1 ≈ Model A F1 (synthetic approaches human baseline)

**Current standings:**
- Model A: 71.39% F1 ✅
- Model B: 70.57% F1 ✅
- Model C: TBD (pending proper synthetic data)

---

## Contact/Session Info
- Working directory: `/Users/arsenispapachristos/Desktop/demo`
- Python: 3.9
- Platform: macOS (Darwin 25.0.0)
- Date: 2026-02-08
