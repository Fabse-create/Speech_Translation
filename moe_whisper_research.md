# MoE + Whisper LoRA Research and Project Analysis

This note captures: (1) a short research summary on MoE and LoRA for Whisper,
(2) an analysis of the current repository state, (3) gaps/risks, and (4)
recommendations for the next steps in the MoE fine-tuning pipeline.

## Research Summary

### Mixture of Experts (MoE)

- MoE uses multiple expert subnetworks and a gating (routing) model that assigns
  each input to one or more experts, often via a softmax over expert logits.
- Practical MoE systems add auxiliary "load balancing" losses to prevent expert
  collapse (a few experts receiving most data) and encourage specialization.
- Soft gating (weighted mixture) is easier to train end-to-end; hard/top-k
  gating is more efficient but needs care for differentiability.

References:
- https://en.wikipedia.org/wiki/Mixture_of_experts
- https://huggingface.co/blog/moe

### Whisper Fine-Tuning with LoRA

- LoRA inserts small low-rank adapters into attention (and sometimes FFN)
  projection layers, keeping most weights frozen to reduce memory and compute.
- Common target modules: attention projections (e.g., `q_proj`, `v_proj`).
- Typical LoRA hyperparameters in ASR: rank r ~ 8–32, alpha ~ 2*r, dropout
  ~0.05–0.1, with AdamW and a small learning rate.
- Mixed precision training is standard for large Whisper models.

References:
- https://aws.amazon.com/blogs/machine-learning/fine-tune-whisper-models-on-amazon-sagemaker-with-lora/
- https://www.mdpi.com/2076-3417/15/24/13090
- https://github.com/Vaibhavs10/fast-whisper-finetuning

### MoE + LoRA for Speech / ASR

- Recent work shows MoE with LoRA adapters can improve ASR in domain- or
  accent-shifted settings by letting experts specialize.

References:
- https://arxiv.org/abs/2404.15159 (MixLoRA)
- https://arxiv.org/abs/2505.20006 (Mixture of LoRA Experts for ASR)

## Repository Analysis (Current State)

### What is already implemented

- Embedding extraction from Whisper v2/v3 (Hugging Face as the single source):
  - `Models/Whisper/whisper_v2.py`, `Models/Whisper/whisper_v3.py`
  - `Data/embedding_extraction.py`
- Clustering pipeline on extracted embeddings:
  - `Data/clustering.py` + `Clustering_Algorithms/*`
- Gating model:
  - `Models/Gating_Model/gating_model.py`
  - `Training_Scripts/gating_model_pre_training.py`
- Joint MoE training:
  - `Training_Scripts/asr_training.py`
- Data loader:
  - `Data/datapreprocessing.py` provides `WhisperDataLoader`
- Project plan in `README.md` aligned with MoE pipeline stages.

### Missing or incomplete pieces

- No MT-aware joint training script yet (`Training_Scripts/asr_mt_training.py`).
- Evaluation loop for ASR (WER/CER) is not yet integrated into training scripts.

## Potential Mismatches / Failure Points

- **Embedding mismatch**: clustering and training must share the exact same
  Whisper implementation. We standardize on Hugging Face Whisper for all
  embeddings, routing, and training so gate assignments stay consistent.
- **Cluster size imbalance**: some HDBSCAN clusters can be very small. Experts
  trained on tiny subsets will overfit or underperform.
- **Noise cluster**: HDBSCAN soft output can include an extra "noise" dimension;
  if `num_experts` does not include that, expert assignment can break.
- **Transcription source**: dataset uses prompt transcripts; for impaired speech,
  the spoken content can deviate. Loss may penalize legitimate deviations. This
  can be mitigated by (1) filtering samples where ASR confidence is low,
  (2) using a softer training target (e.g., include alternate references if
  available), or (3) including a small amount of aligned manual transcripts.
- **Resource constraints**: full Whisper-v2 training is expensive. LoRA is needed
  to keep memory usage practical, but dependencies are not yet present.
- **Routing collapse**: without load-balancing losses in joint training, gate
  can overuse a subset of experts.

## Recommended Decisions

1. **Expert pre-training fraction**: start with 30% of each expert's cluster
   data for pre-training (configurable). This keeps data for joint training and
   validation.
2. **Expert count**: align `num_experts` with clustering outputs; if HDBSCAN
   includes a noise cluster, drop it or add a dedicated expert for noise.
3. **Top-k routing**: limit expert usage per sample (e.g., top-2 or top-3) to
   stabilize pre-training when using soft clustering outputs.
4. **LoRA targets**: start with `q_proj` and `v_proj` only; expand to FFN
   projections if needed.
5. **Load balancing**: for joint training, add an auxiliary loss that penalizes
   expert usage imbalance.
6. **Embedding consistency**: keep a single Whisper source (HF) for embeddings,
   routing, and training.

## Implementation Notes for `expert_pre_training.py`

The new implementation:

- Loads dataset samples via `WhisperDataLoader`.
- Assigns each sample to a top-k set of experts using soft clustering labels
  (or a frozen gating model + precomputed embeddings).
- Fine-tunes each expert sequentially using LoRA adapters on the assigned data.
- Saves each expert adapter in `checkpoints/experts/expert_{id}`.
- Uses Hugging Face `transformers` + `peft`. If they are not installed, the
  script raises a clear error describing required dependencies.

## Single Whisper Source Decision

The project now standardizes on **Hugging Face Whisper** as the only Whisper
implementation used for:

- Embedding extraction
- Gating model inputs
- Expert and joint training

This removes any inconsistency between OpenAI Whisper embeddings and HF Whisper
training behavior.
