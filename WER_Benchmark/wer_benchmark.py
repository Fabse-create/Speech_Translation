import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Data.datapreprocessing import WhisperDataLoader
from Evaluation.evaluate_WER import WERScorer
from Models.Gating_Model.gating_model import GatingModel
from utils.audio import load_audio
from utils.load_config import load_config

try:  # Optional dependency from Hugging Face
    from transformers import WhisperForConditionalGeneration, WhisperModel, WhisperProcessor
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "transformers is required for WER benchmarking. "
        "Install with: pip install transformers"
    ) from exc


def _batched(items: List[Dict[str, str]], batch_size: int) -> Iterable[List[Dict[str, str]]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


class WERTracker:
    def __init__(self) -> None:
        self.scorer = WERScorer(normalize=True)
        self.references: List[str] = []
        self.hypotheses: List[str] = []
        self.by_etiology: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: {"refs": [], "hyps": []}
        )

    def add(self, reference: str, hypothesis: str, etiology: str) -> None:
        self.references.append(reference)
        self.hypotheses.append(hypothesis)
        self.by_etiology[etiology]["refs"].append(reference)
        self.by_etiology[etiology]["hyps"].append(hypothesis)

    def results(self) -> Dict[str, object]:
        total_wer = self.scorer.corpus_wer(self.references, self.hypotheses)
        per_etiology: Dict[str, Dict[str, object]] = {}
        for etiology, data in self.by_etiology.items():
            wer = self.scorer.corpus_wer(data["refs"], data["hyps"])
            per_etiology[etiology] = {
                "wer": float(wer),
                "samples": len(data["refs"]),
            }
        return {
            "total_wer": float(total_wer),
            "samples": len(self.references),
            "per_etiology": per_etiology,
        }


def _prepare_samples(
    dataloader_config: str,
    dataset_root: Optional[str],
    split: str,
    percent: float,
    sampling: str,
    seed: int,
    max_samples: Optional[int],
) -> List[Dict[str, str]]:
    override: Dict[str, object] = {
        "split": split,
        "percent": percent,
        "sampling": sampling,
        "seed": seed,
        "max_samples": max_samples,
    }
    if dataset_root:
        override["dataset_root"] = dataset_root

    loader = WhisperDataLoader(
        config_path=dataloader_config,
        mode="default",
        config=override,
    )
    samples = [sample for sample in loader.sample() if sample.get("prompt")]
    if not samples:
        raise ValueError("No samples with transcripts found for evaluation.")
    return samples


def _configure_generation(
    model: WhisperForConditionalGeneration,
    language: Optional[str],
    task: Optional[str],
) -> None:
    generation_config = getattr(model, "generation_config", model.config)
    if language:
        generation_config.language = language
    if task:
        generation_config.task = task
    if hasattr(generation_config, "forced_decoder_ids"):
        generation_config.forced_decoder_ids = None


def _forward_model(model: torch.nn.Module, **kwargs: torch.Tensor) -> Any:
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "forward"):
        return base_model(**kwargs)
    return model(**kwargs)


def _safe_torch_load(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _resolve_decoder_start_token_id(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
) -> int:
    generation_config = getattr(model, "generation_config", model.config)
    decoder_start_token_id = getattr(
        generation_config, "decoder_start_token_id", None
    )
    if decoder_start_token_id is None:
        decoder_start_token_id = getattr(model.config, "decoder_start_token_id", None)
    if decoder_start_token_id is None:
        decoder_start_token_id = getattr(model.config, "bos_token_id", None)
    if decoder_start_token_id is None and processor is not None:
        decoder_start_token_id = processor.tokenizer.bos_token_id
    if decoder_start_token_id is None:
        raise ValueError("Missing decoder_start_token_id for mixture decoding.")
    return int(decoder_start_token_id)


def _resolve_eos_token_id(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
) -> int:
    generation_config = getattr(model, "generation_config", model.config)
    eos_token_id = getattr(generation_config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0]
    if eos_token_id is None and processor is not None:
        eos_token_id = processor.tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Missing eos_token_id for mixture decoding.")
    return int(eos_token_id)


def _resolve_max_length(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
) -> int:
    generation_config = getattr(model, "generation_config", model.config)
    max_length = getattr(generation_config, "max_length", None)
    if max_length is None:
        max_length = getattr(model.config, "max_length", None)
    if max_length is None and processor is not None:
        max_length = processor.tokenizer.model_max_length
    if max_length is None or max_length <= 0 or max_length > 2048:
        max_length = 448
    return int(max_length)


def _trim_to_eos(sequences: torch.Tensor, eos_token_id: int) -> List[List[int]]:
    trimmed: List[List[int]] = []
    for seq in sequences.tolist():
        if eos_token_id in seq:
            eos_index = seq.index(eos_token_id)
            seq = seq[: eos_index + 1]
        trimmed.append(seq)
    return trimmed


def _decode_topk_mixture(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    input_features: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    use_lora: bool,
) -> List[str]:
    batch_size = input_features.size(0)
    device = input_features.device
    decoder_start_token_id = _resolve_decoder_start_token_id(model, processor)
    eos_token_id = _resolve_eos_token_id(model, processor)
    max_length = _resolve_max_length(model, processor)
    vocab_size = int(getattr(model.config, "vocab_size", 0)) or 51865

    decoder_input_ids = torch.full(
        (batch_size, 1),
        decoder_start_token_id,
        device=device,
        dtype=torch.long,
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    num_experts = int(topk_indices.max().item() + 1) if topk_indices.numel() > 0 else 0
    weight_matrix = torch.zeros(
        batch_size, num_experts, device=device, dtype=topk_weights.dtype
    )
    weight_matrix.scatter_(1, topk_indices, topk_weights)
    past_by_expert: Dict[int, Optional[Tuple[torch.Tensor, ...]]] = {}

    for _ in range(max_length - 1):
        combined_logits = torch.zeros(batch_size, vocab_size, device=device)
        for expert_id in range(num_experts):
            weights = weight_matrix[:, expert_id]
            if not torch.any(weights > 0):
                continue
            if use_lora:
                model.set_adapter(f"expert_{expert_id}")
            mask = weights > 0
            decoder_input = (
                decoder_input_ids[mask]
                if past_by_expert.get(expert_id) is None
                else decoder_input_ids[mask, -1:].contiguous()
            )
            outputs = _forward_model(
                model,
                input_features=input_features[mask],
                attention_mask=attention_mask[mask] if attention_mask is not None else None,
                decoder_input_ids=decoder_input,
                use_cache=True,
                past_key_values=past_by_expert.get(expert_id),
            )
            past_by_expert[expert_id] = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            combined_logits[mask] += weights[mask].unsqueeze(-1) * logits

        if finished.any():
            min_val = torch.finfo(combined_logits.dtype).min
            combined_logits[finished] = min_val
            combined_logits[finished, eos_token_id] = 0.0

        next_tokens = torch.argmax(combined_logits, dim=-1)
        decoder_input_ids = torch.cat(
            [decoder_input_ids, next_tokens.unsqueeze(-1)], dim=-1
        )
        finished = finished | (next_tokens == eos_token_id)
        if finished.all():
            break

    trimmed = _trim_to_eos(decoder_input_ids, eos_token_id)
    return processor.batch_decode(trimmed, skip_special_tokens=True)


def _compute_gating_probs(
    gate_logits: torch.Tensor, temperature: float, min_prob: float = 0.0
) -> torch.Tensor:
    safe_temp = max(1e-4, float(temperature))
    gate_probs = torch.softmax(gate_logits / safe_temp, dim=-1)
    if min_prob > 0:
        gate_probs = torch.clamp(gate_probs, min=min_prob)
        gate_probs = gate_probs / gate_probs.sum(dim=-1, keepdim=True)
    return gate_probs


def _run_baseline(
    samples: List[Dict[str, str]],
    model_name: str,
    device: torch.device,
    batch_size: int,
    language: Optional[str],
    task: Optional[str],
) -> Dict[str, object]:
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    _configure_generation(model, language, task)

    tracker = WERTracker()
    with torch.no_grad():
        for batch in _batched(samples, batch_size):
            audio_list = [load_audio(sample["wav_path"]) for sample in batch]
            inputs = processor(
                audio_list,
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = inputs.input_features.to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            generated_ids = model.generate(
                input_features=input_features, attention_mask=attention_mask
            )
            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
            for sample, pred in zip(batch, preds):
                tracker.add(
                    reference=sample["prompt"],
                    hypothesis=pred,
                    etiology=sample.get("etiology", "Unknown"),
                )

    return tracker.results()


def _resolve_finetuned_dir(base_dir: Path) -> Path:
    candidates = [
        base_dir,
        base_dir / "best",
        Path("checkpoints/asr"),
        Path("checkpoints/asr/best"),
    ]
    for candidate in candidates:
        if (candidate / "gating_model.pt").exists():
            return candidate
    return base_dir


def _load_finetuned_bundle(
    asr_config_path: str,
    fine_tuned_dir: Path,
    device: torch.device,
) -> Dict[str, object]:
    config = load_config(asr_config_path)
    model_name = config.get("model_name", "openai/whisper-large-v2")
    num_experts = int(config.get("num_experts", 8))
    use_lora = bool(config.get("use_lora", True))
    lora_cfg = config.get("lora", {})

    processor = WhisperProcessor.from_pretrained(model_name)
    embedding_model = WhisperModel.from_pretrained(model_name).to(device).eval()
    asr_model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    _configure_generation(
        asr_model, config.get("language"), config.get("task")
    )
    asr_model.eval()

    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "peft is required for loading LoRA experts. "
                "Install with: pip install peft"
            ) from exc

        lora_config = LoraConfig(
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("alpha", 32)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            target_modules=list(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )

        # Support older peft versions without adapter_name in get_peft_model.
        try:
            asr_model = get_peft_model(asr_model, lora_config, adapter_name="expert_0")
        except TypeError:
            asr_model = get_peft_model(asr_model, lora_config)

        if not hasattr(asr_model, "load_adapter"):
            raise RuntimeError(
                "peft version is too old (missing load_adapter). "
                "Please upgrade: pip install -U peft"
            )

        # Helper function to find adapter directory (handles both flat and nested structures)
        def _find_adapter_dir(base_dir: Path, expert_id: int) -> Optional[Path]:
            """Find adapter directory, handling both flat and nested structures.
            
            Checks:
            1. base_dir/expert_X/adapter_config.json (flat)
            2. base_dir/expert_X/expert_X/adapter_config.json (nested)
            3. Recursively searches for adapter_config.json in expert_X/
            """
            expert_dir = base_dir / f"expert_{expert_id}"
            if not expert_dir.exists():
                return None
            
            # Check flat structure
            if (expert_dir / "adapter_config.json").exists():
                return expert_dir
            
            # Check nested structure (expert_X/expert_X/)
            nested_dir = expert_dir / f"expert_{expert_id}"
            if nested_dir.exists() and (nested_dir / "adapter_config.json").exists():
                return nested_dir
            
            # Recursively search for adapter_config.json
            for path in expert_dir.rglob("adapter_config.json"):
                return path.parent
            
            return None
        
        adapter_dirs = [_find_adapter_dir(fine_tuned_dir, i) for i in range(num_experts)]
        adapter_dirs = [d for d in adapter_dirs if d is not None]
        
        if not adapter_dirs:
            raise FileNotFoundError(
                "No LoRA adapters found in fine-tuned directory. "
                f"Expected adapter_config.json under {fine_tuned_dir}/expert_*/ "
                f"(or {fine_tuned_dir}/expert_*/expert_*/ for nested structure). "
                "If you trained with the pipeline, set --fine-tuned-dir to checkpoints/asr."
            )
        
        for expert_id in range(num_experts):
            adapter_dir = _find_adapter_dir(fine_tuned_dir, expert_id)
            if adapter_dir is not None:
                # Convert to absolute path to avoid PEFT treating it as a Hub repo ID
                adapter_path = adapter_dir.resolve()
                asr_model.load_adapter(str(adapter_path), adapter_name=f"expert_{expert_id}")

        # Set default adapter if expert_0 exists
        if hasattr(asr_model, "set_adapter"):
            expert_0_dir = _find_adapter_dir(fine_tuned_dir, 0)
            if expert_0_dir is not None:
                asr_model.set_adapter("expert_0")

    gating_config_path = config.get("gating_model_config", "Config/gating_model_config.json")
    gating_model = GatingModel(config_path=gating_config_path).to(device)
    gating_checkpoint = fine_tuned_dir / "gating_model.pt"
    if not gating_checkpoint.exists():
        raise FileNotFoundError(
            f"Missing gating model checkpoint: {gating_checkpoint}"
        )
    gating_model.load_state_dict(_safe_torch_load(gating_checkpoint, device))
    gating_model.eval()

    routing_temperature = float(
        config.get(
            "routing_temperature_min",
            config.get("routing_temperature_end", 1.0),
        )
    )

    return {
        "processor": processor,
        "embedding_model": embedding_model,
        "asr_model": asr_model,
        "gating_model": gating_model,
        "num_experts": num_experts,
        "use_lora": use_lora,
        "routing_temperature": routing_temperature,
    }


def _run_finetuned(
    samples: List[Dict[str, str]],
    asr_config_path: str,
    fine_tuned_dir: Path,
    device: torch.device,
    batch_size: int,
    gating_output_path: Optional[Path] = None,
    moe_top_k: int = 1,
    moe_mixture: bool = False,
) -> Dict[str, object]:
    bundle = _load_finetuned_bundle(asr_config_path, fine_tuned_dir, device)
    processor = bundle["processor"]
    embedding_model = bundle["embedding_model"]
    asr_model = bundle["asr_model"]
    gating_model = bundle["gating_model"]
    use_lora = bundle["use_lora"]
    num_experts = int(bundle["num_experts"])
    routing_temperature = float(bundle["routing_temperature"])
    effective_top_k = max(1, min(int(moe_top_k), num_experts))

    tracker = WERTracker()
    expert_usage_overall: Dict[int, int] = defaultdict(int)
    expert_usage_by_etiology: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    expert_topk_usage_overall: Dict[int, int] = defaultdict(int)
    expert_topk_usage_by_etiology: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    gate_prob_sum = torch.zeros(num_experts, dtype=torch.float64)
    gate_prob_count = 0
    gate_prob_sum_by_etiology: Dict[str, torch.Tensor] = defaultdict(
        lambda: torch.zeros(num_experts, dtype=torch.float64)
    )
    gate_prob_count_by_etiology: Dict[str, int] = defaultdict(int)
    gate_entropy_sum = 0.0
    gating_handle = None
    if gating_output_path is not None:
        gating_output_path.parent.mkdir(parents=True, exist_ok=True)
        gating_handle = gating_output_path.open("w", encoding="utf-8")
    with torch.no_grad():
        try:
            for batch in _batched(samples, batch_size):
                audio_list = [load_audio(sample["wav_path"]) for sample in batch]
                inputs = processor(
                    audio_list,
                    sampling_rate=16000,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                input_features = inputs.input_features.to(device)
                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                encoder_outputs = embedding_model.encoder(
                    input_features, attention_mask=attention_mask
                )
                pooled = encoder_outputs.last_hidden_state.mean(dim=1)
                gate_probs = _compute_gating_probs(
                    gating_model(pooled),
                    temperature=routing_temperature,
                    min_prob=0.0,
                )
                k = max(1, min(effective_top_k, gate_probs.size(-1)))
                topk_values, topk_indices = torch.topk(gate_probs, k=k, dim=-1)
                topk_weights = topk_values / topk_values.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)
                top1_indices = topk_indices[:, 0]
                gate_probs_cpu = gate_probs.detach().cpu().double()
                top1_cpu = top1_indices.detach().cpu().tolist()
                topk_cpu = topk_indices.detach().cpu().tolist()

                gate_prob_sum += gate_probs_cpu.sum(dim=0)
                gate_prob_count += gate_probs_cpu.size(0)
                gate_entropy = -(gate_probs_cpu * torch.log(gate_probs_cpu + 1e-8)).sum(dim=-1)
                gate_entropy_sum += float(gate_entropy.sum().item())

                for idx, sample in enumerate(batch):
                    etiology = sample.get("etiology", "Unknown")
                    gate_prob_sum_by_etiology[etiology] += gate_probs_cpu[idx]
                    gate_prob_count_by_etiology[etiology] += 1
                    for expert_id in topk_cpu[idx]:
                        expert_topk_usage_overall[expert_id] += 1
                        expert_topk_usage_by_etiology[etiology][expert_id] += 1
                    if gating_handle is not None:
                        payload = {
                            "id": sample.get("id"),
                            "etiology": etiology,
                            "gate_probs": [float(x) for x in gate_probs_cpu[idx].tolist()],
                            "top1_expert": int(top1_cpu[idx]),
                        }
                        if k > 1:
                            payload["topk_experts"] = [int(x) for x in topk_cpu[idx]]
                        gating_handle.write(json.dumps(payload) + "\n")

                if moe_mixture and k > 1:
                    predictions = _decode_topk_mixture(
                        model=asr_model,
                        processor=processor,
                        input_features=input_features,
                        attention_mask=attention_mask,
                        topk_indices=topk_indices,
                        topk_weights=topk_weights,
                        use_lora=use_lora,
                    )
                else:
                    predictions = [""] * input_features.size(0)
                    for expert_id in torch.unique(top1_indices).tolist():
                        mask = top1_indices == expert_id
                        if not mask.any():
                            continue
                        if use_lora:
                            asr_model.set_adapter(f"expert_{expert_id}")
                        feats = input_features[mask]
                        feats_mask = (
                            attention_mask[mask] if attention_mask is not None else None
                        )
                        generated_ids = asr_model.generate(
                            input_features=feats, attention_mask=feats_mask
                        )
                        preds = processor.batch_decode(
                            generated_ids, skip_special_tokens=True
                        )
                        for idx, pred in zip(torch.where(mask)[0].tolist(), preds):
                            predictions[idx] = pred

                for sample, pred, expert_id in zip(
                    batch, predictions, top1_cpu
                ):
                    etiology = sample.get("etiology", "Unknown")
                    tracker.add(
                        reference=sample["prompt"],
                        hypothesis=pred,
                        etiology=etiology,
                    )
                    expert_usage_overall[expert_id] += 1
                    expert_usage_by_etiology[etiology][expert_id] += 1
        finally:
            if gating_handle is not None:
                gating_handle.close()

    results = tracker.results()
    decoding_mode = "top1"
    if moe_mixture and effective_top_k > 1:
        decoding_mode = f"top{int(effective_top_k)}_mixture"
    results["expert_usage"] = {
        "overall": {str(i): int(expert_usage_overall.get(i, 0)) for i in range(num_experts)},
        "per_etiology": {
            etiology: {
                str(i): int(counts.get(i, 0)) for i in range(num_experts)
            }
            for etiology, counts in sorted(expert_usage_by_etiology.items())
        },
    }
    if effective_top_k > 1:
        results["expert_topk_usage"] = {
            "overall": {str(i): int(expert_topk_usage_overall.get(i, 0)) for i in range(num_experts)},
            "per_etiology": {
                etiology: {
                    str(i): int(counts.get(i, 0)) for i in range(num_experts)
                }
                for etiology, counts in sorted(expert_topk_usage_by_etiology.items())
            },
        }
    gate_prob_mean = (gate_prob_sum / max(1, gate_prob_count)).tolist()
    results["gating_probabilities"] = {
        "temperature": float(routing_temperature),
        "mean_overall": {str(i): float(gate_prob_mean[i]) for i in range(num_experts)},
        "mean_entropy": float(gate_entropy_sum / max(1, gate_prob_count)),
        "per_etiology": {
            etiology: {
                str(i): float(
                    (
                        gate_prob_sum_by_etiology[etiology]
                        / max(1, gate_prob_count_by_etiology[etiology])
                    )[i]
                )
                for i in range(num_experts)
            }
            for etiology in sorted(gate_prob_sum_by_etiology.keys())
        },
        "per_sample_path": str(gating_output_path) if gating_output_path else None,
    }
    results["moe_decoding"] = decoding_mode
    results["moe_top_k"] = int(effective_top_k)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark WER for Whisper v2/v3 and fine-tuned MoE ASR."
    )
    parser.add_argument(
        "--models",
        default="v2,v3,finetuned",
        help="Comma-separated list: v2,v3,finetuned",
    )
    parser.add_argument(
        "--dataloader-config",
        default="Config/dataloader_config.json",
        help="Path to dataloader config.",
    )
    parser.add_argument("--dataset-root", default=None, help="Override dataset root.")
    parser.add_argument("--split", default="Dev", choices=["Train", "Dev"])
    parser.add_argument("--percent", type=float, default=20.0)
    parser.add_argument(
        "--sampling",
        default="stratified",
        choices=["random", "stratified"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--asr-config",
        default="Config/asr_training.json",
        help="Path to ASR config for fine-tuned model loading.",
    )
    parser.add_argument(
        "--fine-tuned-dir",
        default="checkpoints/asr",
        help="Directory containing gating_model.pt and expert adapters.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language for forced decoder ids (e.g., en).",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Optional task for forced decoder ids (e.g., transcribe).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path for results.",
    )
    parser.add_argument(
        "--gating-output",
        default=None,
        help="Optional JSONL output path for per-sample gating probabilities.",
    )
    parser.add_argument(
        "--moe-top-k",
        type=int,
        default=1,
        help="Top-k experts to use for mixture decoding (finetuned only).",
    )
    parser.add_argument(
        "--moe-mixture",
        action="store_true",
        help="Enable top-k mixture decoding for finetuned model.",
    )

    args = parser.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    samples = _prepare_samples(
        dataloader_config=args.dataloader_config,
        dataset_root=args.dataset_root,
        split=args.split,
        percent=args.percent,
        sampling=args.sampling,
        seed=args.seed,
        max_samples=args.max_samples,
    )

    # Align baseline decoding with ASR config if no explicit language/task overrides.
    asr_config = load_config(args.asr_config)
    language = args.language if args.language is not None else asr_config.get("language")
    task = args.task if args.task is not None else asr_config.get("task")

    models = [name.strip().lower() for name in args.models.split(",") if name.strip()]
    results: Dict[str, object] = {
        "settings": {
            "split": args.split,
            "percent": args.percent,
            "sampling": args.sampling,
            "seed": args.seed,
            "max_samples": args.max_samples,
            "samples_used": len(samples),
            "moe_top_k": args.moe_top_k,
            "moe_decoding": "topk_mixture" if args.moe_mixture else "top1",
        }
    }

    if "v2" in models:
        results["whisper_v2"] = _run_baseline(
            samples=samples,
            model_name="openai/whisper-large-v2",
            device=device,
            batch_size=args.batch_size,
            language=language,
            task=task,
        )

    if "v3" in models:
        results["whisper_v3"] = _run_baseline(
            samples=samples,
            model_name="openai/whisper-large-v3",
            device=device,
            batch_size=args.batch_size,
            language=language,
            task=task,
        )

    if "finetuned" in models:
        fine_tuned_dir = _resolve_finetuned_dir(Path(args.fine_tuned_dir))
        results["finetuned_asr"] = _run_finetuned(
            samples=samples,
            asr_config_path=args.asr_config,
            fine_tuned_dir=fine_tuned_dir,
            device=device,
            batch_size=args.batch_size,
            gating_output_path=Path(args.gating_output) if args.gating_output else None,
            moe_top_k=args.moe_top_k,
            moe_mixture=args.moe_mixture,
        )

    print(json.dumps(results, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
