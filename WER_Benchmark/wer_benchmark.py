import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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


def _set_forced_decoder_ids(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    language: Optional[str],
    task: Optional[str],
) -> None:
    if language or task:
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, task=task
        )


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
    _set_forced_decoder_ids(model, processor, language, task)

    tracker = WERTracker()
    with torch.no_grad():
        for batch in _batched(samples, batch_size):
            audio_list = [load_audio(sample["wav_path"]) for sample in batch]
            inputs = processor(
                audio_list, sampling_rate=16000, return_tensors="pt"
            )
            input_features = inputs.input_features.to(device)
            generated_ids = model.generate(input_features=input_features)
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
        Path("Runs/full/asr"),
        Path("Runs/full/asr/best"),
        Path("Runs/quick/asr"),
        Path("Runs/quick/asr/best"),
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
    _set_forced_decoder_ids(
        asr_model, processor, config.get("language"), config.get("task")
    )

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

        adapter_dirs = [fine_tuned_dir / f"expert_{i}" for i in range(num_experts)]
        if not any((adapter_dir / "adapter_config.json").exists() for adapter_dir in adapter_dirs):
            raise FileNotFoundError(
                "No LoRA adapters found in fine-tuned directory. "
                f"Expected adapter_config.json under {fine_tuned_dir}/expert_*/. "
                "If you trained with the pipeline, set --fine-tuned-dir to Runs/full/asr "
                "(or Runs/quick/asr)."
            )
        for expert_id, adapter_dir in enumerate(adapter_dirs):
            if (adapter_dir / "adapter_config.json").exists():
                asr_model.load_adapter(str(adapter_dir), adapter_name=f"expert_{expert_id}")

        if hasattr(asr_model, "set_adapter") and (fine_tuned_dir / "expert_0").exists():
            asr_model.set_adapter("expert_0")

    gating_config_path = config.get("gating_model_config", "Config/gating_model_config.json")
    gating_model = GatingModel(config_path=gating_config_path).to(device)
    gating_checkpoint = fine_tuned_dir / "gating_model.pt"
    if not gating_checkpoint.exists():
        raise FileNotFoundError(
            f"Missing gating model checkpoint: {gating_checkpoint}"
        )
    gating_model.load_state_dict(torch.load(gating_checkpoint, map_location=device))
    gating_model.eval()

    return {
        "processor": processor,
        "embedding_model": embedding_model,
        "asr_model": asr_model,
        "gating_model": gating_model,
        "num_experts": num_experts,
        "use_lora": use_lora,
    }


def _run_finetuned(
    samples: List[Dict[str, str]],
    asr_config_path: str,
    fine_tuned_dir: Path,
    device: torch.device,
    batch_size: int,
) -> Dict[str, object]:
    bundle = _load_finetuned_bundle(asr_config_path, fine_tuned_dir, device)
    processor = bundle["processor"]
    embedding_model = bundle["embedding_model"]
    asr_model = bundle["asr_model"]
    gating_model = bundle["gating_model"]
    use_lora = bundle["use_lora"]
    num_experts = int(bundle["num_experts"])

    tracker = WERTracker()
    expert_usage_overall: Dict[int, int] = defaultdict(int)
    expert_usage_by_etiology: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    with torch.no_grad():
        for batch in _batched(samples, batch_size):
            audio_list = [load_audio(sample["wav_path"]) for sample in batch]
            inputs = processor(
                audio_list, sampling_rate=16000, return_tensors="pt"
            )
            input_features = inputs.input_features.to(device)

            encoder_outputs = embedding_model.encoder(input_features)
            pooled = encoder_outputs.last_hidden_state.mean(dim=1)
            gate_probs = torch.softmax(gating_model(pooled), dim=-1)
            top1_indices = torch.argmax(gate_probs, dim=-1)

            predictions = [""] * input_features.size(0)
            for expert_id in torch.unique(top1_indices).tolist():
                mask = top1_indices == expert_id
                if not mask.any():
                    continue
                if use_lora:
                    asr_model.set_adapter(f"expert_{expert_id}")
                feats = input_features[mask]
                generated_ids = asr_model.generate(input_features=feats)
                preds = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                for idx, pred in zip(torch.where(mask)[0].tolist(), preds):
                    predictions[idx] = pred

            for sample, pred, expert_id in zip(
                batch, predictions, top1_indices.tolist()
            ):
                etiology = sample.get("etiology", "Unknown")
                tracker.add(
                    reference=sample["prompt"],
                    hypothesis=pred,
                    etiology=etiology,
                )
                expert_usage_overall[expert_id] += 1
                expert_usage_by_etiology[etiology][expert_id] += 1

    results = tracker.results()
    results["expert_usage"] = {
        "overall": {str(i): int(expert_usage_overall.get(i, 0)) for i in range(num_experts)},
        "per_etiology": {
            etiology: {
                str(i): int(counts.get(i, 0)) for i in range(num_experts)
            }
            for etiology, counts in sorted(expert_usage_by_etiology.items())
        },
    }
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
        )

    print(json.dumps(results, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
