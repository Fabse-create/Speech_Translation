# Training Analysis

This document summarizes the current training behavior observed in `whisper_3008952.err`,
explains why experts train sequentially, details evaluation logic for the gating model,
experts, and the full MoE system, and shows how to load the models individually or together.

## What Looks Good / Bad / Unexpected

### Looks good
- Embedding extraction completed successfully with 1608 embeddings.
- Gating pretraining shows a steady downward trend in train/val loss and rising val accuracy
  through 20 epochs (up to ~0.74).
- The pipeline advanced through all stages (embedding extraction, clustering, gating training,
  expert pretraining, and ASR training initialization), indicating overall orchestration works.

### Looks bad
- Expert assignment reports **4157/4823 samples missing embeddings**, which means most samples
  were skipped during expert routing. This weakens expert specialization and reduces
  effective training data.
- This mismatch is caused by the data-percentage scaling in full mode: embeddings are only
  0.5% of data when `data_percent=10%`, while experts use 1.5% and ASR uses 10%.

```706:737:Training_Scripts/train_pipeline.py
    if mode == "quick":
        embedding_percent = 100
        ...
    else:
        embedding_percent = 5
        expert_percent = 15
        asr_percent = 100
        ...

    if data_percent is not None:
        scale_factor = data_percent / 100.0
        embedding_percent = embedding_percent * scale_factor
        expert_percent = expert_percent * scale_factor
        asr_percent = asr_percent * scale_factor
```

### Looks unexpected
- With `data_percent=10%`, embeddings drop to **0.5%** while experts/ASR use larger fractions.
  This creates a mismatch between routing data and training data.
- The UMAP warning about `n_jobs` being forced to 1 is expected with a fixed `random_state`
  but is still surprising if you expect parallel clustering speedups.
- Gating validation accuracy is computed as argmax accuracy even though training uses soft
  label distributions; this is a weaker proxy metric for soft targets.

## Why Experts Train Sequentially

Experts are trained one after another by design. The expert pretraining script loops over
each expert id and calls `_train_expert` separately. Each expert gets its own subset of data,
its own training loop, and is saved to `checkpoints/experts/expert_{id}`. This is not a
"round-robin" per-epoch scheme; it is separate training runs per expert.

```728:745:Training_Scripts/expert_pre_training.py
for expert_id, indices in indices_by_expert.items():
    if not indices:
        print(f"\nSkipping expert {expert_id}: no assigned samples.")
        continue
    print(f"\n{'='*50}")
    print(f"Training expert {expert_id} with {len(indices)} samples")
    print(f"{'='*50}")
    _train_expert(
        expert_id=expert_id,
        dataset=dataset,
        indices=indices,
        config=config,
        processor=processor,
        device=device,
    )
```

## Evaluation Logic and Theoretical Foundation

### 1) Gating model (pretraining)

**What the code does**
- The gating model is trained on precomputed embeddings and clustering labels.
- It minimizes KL divergence between predicted logits and soft label distributions.
- Validation computes both loss and argmax accuracy.

```163:219:Training_Scripts/gating_model_pre_training.py
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = nn.KLDivLoss(reduction="batchmean")
...
logits = model(inputs)
log_probs = torch.log_softmax(logits, dim=-1)
loss = loss_fn(log_probs, targets)
...
metrics = evaluate_model(model, val_loader, device)
```

```7:38:Evaluation/evaluate_model_training.py
loss_fn = nn.KLDivLoss(reduction="batchmean")
...
pred = torch.argmax(logits, dim=-1)
true = torch.argmax(targets, dim=-1)
accuracy = correct / max(total_samples, 1)
```

**Theory**
- This is a classic Mixture-of-Experts (MoE) routing setup: the gating network learns to
  output a distribution over experts conditioned on an embedding.
- KL divergence against soft targets is a standard way to match probability distributions.
- MoE routing and load balancing are documented in MoE literature such as Shazeer et al. (2017)
  and Switch Transformers (Fedus et al., 2021).

### 2) Expert pretraining

**What the code does**
- Each expert is trained on a subset of samples routed by the gating model.
- Validation metrics include standard seq2seq loss and WER (word error rate).

```524:581:Training_Scripts/expert_pre_training.py
for epoch in range(1, config.epochs + 1):
    ...
    if val_size > 0:
        val_loss = _evaluate(model, val_loader, device, config.fp16)
        val_wer = _evaluate_wer(model, val_loader, processor, device)
        ...
        print(
            f"Expert {expert_id} epoch {epoch}: "
            f"val_loss={val_loss:.4f} val_wer={val_wer:.4f}"
        )
```

```435:461:Training_Scripts/expert_pre_training.py
generated_ids = model.generate(input_features=input_features)
preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
...
refs = processor.batch_decode(labels, skip_special_tokens=True)
...
return scorer.corpus_wer(references, hypotheses)
```

```25:69:Evaluation/evaluate_WER.py
def corpus_wer(self, references: Iterable[str], hypotheses: Iterable[str]) -> float:
    ...
    total_edits += self._edit_distance(ref_tokens, hyp_tokens)
    total_words += len(ref_tokens)
    return total_edits / total_words
```

**Theory**
- WER is based on Levenshtein edit distance: insertions + deletions + substitutions
  normalized by reference word count.
- This is the standard ASR evaluation metric.

### 3) Full system (gating + experts + Whisper)

**What the code does**
- A frozen Whisper encoder (`WhisperModel`) generates pooled embeddings for the gate.
- The gate selects top-k experts and assigns weights.
- The selected expert adapters are used for each subset of the batch.
- Loss combines per-sample seq2seq loss with a load-balance term.
- Evaluation uses validation loss and WER; test loss/WER are computed at the end.

```281:364:Training_Scripts/asr_training.py
with torch.no_grad():
    encoder_outputs = embedding_model.encoder(input_features)
    pooled = encoder_outputs.last_hidden_state.mean(dim=1)
gate_logits = gating_model(pooled)
gate_probs = torch.softmax(gate_logits, dim=-1)
k = max(1, min(config.top_k_experts, gate_probs.size(-1)))
topk_values, topk_indices = torch.topk(gate_probs, k=k, dim=-1)
topk_weights = topk_values / topk_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
...
main_loss = per_sample_loss.mean()
balance_loss = _load_balance_loss(gate_probs, config.num_experts)
loss = main_loss + config.load_balance_coef * balance_loss
```

```461:549:Training_Scripts/train_pipeline.py
for epoch in range(1, config.epochs + 1):
    train_loss = asr_training._train_epoch(...)
    if len(val_set) > 0:
        val_loss = asr_training._evaluate(...)
        val_wer = asr_training._evaluate_wer(...)
...
if len(test_set) > 0:
    test_loss = asr_training._evaluate(...)
    test_wer = asr_training._evaluate_wer(...)
```

**Theory**
- MoE routing encourages specialization by selecting only the top-k experts per input.
- The load-balance term is a standard MoE trick to avoid expert collapse, discussed in
  Switch Transformers.
- The WER evaluation is identical to expert pretraining (edit-distance-based).

## Model Loading How-To

### 1) Load Whisper v2 / v3 without adaptation

Use Hugging Face Transformers directly:

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model_name = "openai/whisper-large-v2"  # or "openai/whisper-large-v3"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.eval()
```

### 2) Load the gating model alone

```python
import torch
from Models.Gating_Model.gating_model import GatingModel

gating = GatingModel(config_path="Config/gating_model_config.json")
state = torch.load("checkpoints/gating_model/best.pt", map_location="cpu", weights_only=True)
gating.load_state_dict(state)
gating.eval()
```

### 3) Load experts and use them with Whisper

Experts are saved as LoRA adapters in `checkpoints/experts/expert_{id}`. Load the base model
and then load adapters:

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model_name = "openai/whisper-large-v2"
processor = WhisperProcessor.from_pretrained(model_name)
base = WhisperForConditionalGeneration.from_pretrained(model_name)

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(base, lora_config, adapter_name="expert_0")
for expert_id in range(1, 8):
    model.add_adapter(f"expert_{expert_id}", lora_config)

for expert_id in range(8):
    model.load_adapter(f"checkpoints/experts/expert_{expert_id}", adapter_name=f"expert_{expert_id}")
```

If `use_lora` is disabled, experts are saved as full models instead:

```python
model = WhisperForConditionalGeneration.from_pretrained("checkpoints/experts/expert_0")
```

### 4) Load Whisper + gating + experts (full system)

This mirrors the training pipeline and `_evaluate_wer` logic:

```python
import torch
from transformers import WhisperModel, WhisperForConditionalGeneration, WhisperProcessor
from Models.Gating_Model.gating_model import GatingModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "openai/whisper-large-v2"

processor = WhisperProcessor.from_pretrained(model_name)
embedding_model = WhisperModel.from_pretrained(model_name).to(device).eval()
asr_model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

# Load LoRA adapters as shown above
# ...

gating = GatingModel(config_path="Config/gating_model_config.json").to(device)
state = torch.load("checkpoints/gating_model/best.pt", map_location=device, weights_only=True)
gating.load_state_dict(state)
gating.eval()

# Forward pass example (top-1 routing)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
with torch.no_grad():
    encoder_outputs = embedding_model.encoder(inputs.input_features)
    pooled = encoder_outputs.last_hidden_state.mean(dim=1)
    gate_probs = torch.softmax(gating(pooled), dim=-1)
    top1 = torch.argmax(gate_probs, dim=-1)

for expert_id in torch.unique(top1).tolist():
    asr_model.set_adapter(f"expert_{expert_id}")
    mask = top1 == expert_id
    if not mask.any():
        continue
    feats = inputs.input_features[mask]
    generated_ids = asr_model.generate(input_features=feats)
    # decode via processor.batch_decode(...)
```

Notes:
- Training uses top-k routing with weighted losses; evaluation uses top-1 routing for decoding.
- The gating model expects pooled encoder embeddings matching `input_dim` from
  `Config/gating_model_config.json` (default 1280).

## References (Theory and Model Docs)

- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
  (2017) https://arxiv.org/abs/1701.06538
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
  (2021) https://arxiv.org/abs/2101.03961
- Levenshtein, "Binary Codes Capable of Correcting Deletions, Insertions, and Reversals" (1966)
  https://doi.org/10.1016/S0019-9958(66)90018-4
- Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper, 2022)
  https://arxiv.org/abs/2212.04356
- Hugging Face Whisper docs and model cards:
  https://huggingface.co/docs/transformers/model_doc/whisper
  https://huggingface.co/openai/whisper-large-v2
  https://huggingface.co/openai/whisper-large-v3
