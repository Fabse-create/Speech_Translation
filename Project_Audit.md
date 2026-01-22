# Project Audit Report

Date: 2026-01-21
Scope: code, configs, docker files, training/translation scripts, docs

## High-risk issues
- **Hardcoded secrets in repo.** `Translation/translation_deepl.py` contains a DeepL API key in plain text. This is a security risk and will leak if the repo is shared. Use env vars and remove the key from version control.
- **Docker containers that cannot run.** `docker/embedding/Dockerfile` and `docker/clustering/Dockerfile` never copy the repository into the image. The `CMD` points to `Data.*` modules, but `/app` is empty, so these containers will fail immediately.
- **Clustering uses raw variable-length embeddings.** `Data/clustering.py` loads `.npy` files and does `np.asarray(embeddings)` without pooling/flattening. Whisper embeddings are sequence-length x 1280, so this results in object arrays or 3D tensors and will break HDBSCAN / Spectral clustering. This likely prevents clustering from running at all.
- **Potential mismatch between cluster count and `num_experts`.** HDBSCAN produces a variable number of clusters; `Config/gating_model_config.json` fixes `num_experts` to 8. If HDBSCAN finds a different number of clusters or includes a noise column in soft labels, `Training_Scripts/gating_model_pre_training.py` will error or silently train on mismatched targets.

## Medium-risk issues
- **Docker dependency mismatch for embedding extraction.** `docker/embedding/requirements.txt` installs `openai-whisper`, but `Data/embedding_extraction.py` uses the Transformers Whisper classes and `utils/audio` depends on `torchaudio` or `librosa`. Those are missing, so embedding extraction in Docker will fail.
- **Translation scripts are not portable.** `Translation/*.py` and `Translation/down_sampling.py` use absolute Windows paths (`D:\...`). These will fail on other machines and make automation brittle.
- **`Translation/translation.py` likely broken.** It imports `googletrans`, but the comment says `pygoogletranslation`. It also uses `async with Translator()` and `await translator.translate(...)`, which are not supported in the typical `googletrans` API (sync). Expect runtime errors.
- **`asr_mt_training.py`, `evaluate_BLEU.py`, and `evaluate_WER.py` are empty.** These files exist but contain no logic; any pipeline expecting them will fail or do nothing.
- **No top-level dependency list.** There is no root `requirements.txt` or `pyproject.toml`. Only Docker subfolders define dependencies, which leads to inconsistent environments.

## Low-risk / quality issues
- **README and config mismatch.** `README.md` describes the gating model output as 10 experts, but `Config/gating_model_config.json` uses 8. This will confuse users and can cause wrong expectations.
- **`Translation/translation_deepl.py` lacks response validation.** It assumes the API returns exactly one translation per input. If a request is partial or rate-limited, data could be misaligned without warning.
- **Memory heavy steps.** `Training_Scripts/expert_pre_training.py` loads all embeddings into memory (`_load_embeddings`). With many samples, this can exhaust RAM.

## Dependency and version notes
- **Unpinned dependencies.** All `requirements.txt` files omit versions. This makes runs non-reproducible and can break with upstream changes (Transformers/PEFT/HDBSCAN are especially sensitive).
- **Missing deps for translation and MT.** `Translation/translation_opus.py` needs `ctranslate2`; `Translation/translation_deepl.py` needs `requests`; `Translation/translation.py` needs `googletrans`; `MT/train.py` needs `datasets` and `evaluate`. None are listed anywhere.
- **Potential CPU-only limitation in Docker.** Training Docker images use `pytorch/pytorch:2.2.2-cpu`. That’s fine for debugging but will be extremely slow for Whisper training.

## Data / workflow risks
- **Clustering output compatibility.** `Clustering_Algorithms/hdbscan.py` can add a noise column to soft labels. `Training_Scripts/expert_pre_training.py` trims this only if `drop_noise` is true and only if the soft vector length equals `num_experts + 1`. If HDBSCAN cluster count differs from `num_experts`, routing assignments will be inconsistent.
- **Embedding naming assumptions.** Embedding filenames include the original `.wav` name (`something.wav.npy`). This is consistent internally, but downstream tooling must always use the same ID style or lookups will fail.

## Suggested fixes (ordered by impact)
1. Remove the DeepL API key from `Translation/translation_deepl.py`; read it from `DEEPL_AUTH_KEY` env only.
2. Add `COPY . /app` to `docker/embedding/Dockerfile` and `docker/clustering/Dockerfile`.
3. Pool/flatten embeddings before clustering in `Data/clustering.py` (e.g., mean across time, matching the gating model input).
4. Align `num_experts` with actual clustering output (or enforce a fixed number of clusters) and document it.
5. Add a root dependency file (or per-module requirements) and pin versions.
6. Replace hardcoded paths with config-driven or CLI arguments across `Translation/`.
7. Fix or remove empty scripts, or explicitly mark them as stubs.
