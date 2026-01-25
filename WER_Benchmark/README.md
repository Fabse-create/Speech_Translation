# WER Benchmark (v2, v3, fine-tuned MoE)

This folder contains a standalone script to benchmark Word Error Rate (WER) for:

- Whisper v2 (untuned)
- Whisper v3 (untuned)
- Your fine-tuned MoE ASR (gating + experts)

It reports:

- **Total WER** across all samples
- **WER per etiology/impairment** (based on the dataset `Etiology` metadata)

## Quick Start

```bash
python WER_Benchmark/wer_benchmark.py \
  --models v2,v3,finetuned \
  --split Dev \
  --percent 20 \
  --sampling stratified \
  --max-samples 1000 \
  --batch-size 2 \
  --device cuda \
  --output WER_Benchmark/results.json
```

### Notes on representativeness
- The script defaults to **stratified sampling**, which preserves the etiology mix.
- For a representative estimate, use **10–30%** of the Dev split or at least **300–1000** samples.
- Increase `--max-samples` for more stable per-etiology scores if you have enough data.

## Fine-tuned model loading
The fine-tuned model is expected to live under `checkpoints/asr` (or `checkpoints/asr/best`):

```
checkpoints/asr/
  gating_model.pt
  expert_0/
  expert_1/
  ...
```

If you store it elsewhere, set:

```bash
--fine-tuned-dir path/to/your/asr_output
```

## Output Format
The script prints JSON to stdout and optionally writes it to `--output`.

Example:

```json
{
  "settings": {
    "split": "Dev",
    "percent": 20,
    "sampling": "stratified",
    "seed": 42,
    "max_samples": 1000,
    "samples_used": 482
  },
  "whisper_v2": {
    "total_wer": 0.43,
    "samples": 482,
    "per_etiology": {
      "Spastic": { "wer": 0.51, "samples": 120 },
      "Ataxic": { "wer": 0.39, "samples": 95 }
    }
  },
  "whisper_v3": {
    "total_wer": 0.39,
    "samples": 482,
    "per_etiology": {
      "Spastic": { "wer": 0.48, "samples": 120 },
      "Ataxic": { "wer": 0.36, "samples": 95 }
    }
  },
  "finetuned_asr": {
    "total_wer": 0.28,
    "samples": 482,
    "per_etiology": {
      "Spastic": { "wer": 0.31, "samples": 120 },
      "Ataxic": { "wer": 0.24, "samples": 95 }
    }
  }
}
```

## What scores to expect (realistic ranges)

These are rough ranges for impaired speech; actual values depend on dataset size,
microphone quality, and language variety:

- **Whisper v2 (untuned)**: ~0.30–0.70 WER
- **Whisper v3 (untuned)**: ~0.25–0.65 WER (often slightly better than v2)
- **Fine-tuned MoE**: ~0.15–0.45 WER if trained on the same domain; smaller gains if
  there is a domain mismatch or if embeddings/training data are sparse.

If you see WER higher than 0.70, data quality or transcript alignment is often the cause.

## Tips
- Use the **same Dev split** each time for fair comparisons.
- Keep `--seed` fixed.
- If per-etiology counts are low, increase `--percent` or `--max-samples`.

## Dependencies

The script uses your existing project dependencies plus:

- `transformers` (Whisper models)
- `peft` (for LoRA experts)
- `torchaudio` or `librosa` (audio loading)

Install as needed:

```bash
pip install transformers peft torchaudio
```
