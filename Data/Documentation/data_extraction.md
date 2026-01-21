# Embedding Extraction and Sampling

## What `embedding_extraction.py` does

`Data/embedding_extraction.py` loads samples via `WhisperDataLoader`, extracts
Whisper embeddings (v2 or v3), saves each embedding as a `.npy` file, and writes
`mapping.json` so you can trace each embedding back to its original `.wav` file.

## How to run embedding extraction

```bash
python Data/embedding_extraction.py
```

If `ffmpeg` is not installed locally, you can use the Docker container instead:

```bash
docker build -f docker/embedding/Dockerfile -t embedding-extractor .
docker run --rm -v %cd%:/app embedding-extractor
```

PowerShell variant:

```bash
docker run --rm -v ${PWD}:/app embedding-extractor
```

## Docker: choose what to sample

The Docker container uses the same `Config/embedding_extraction.json` file.
To control what is sampled, edit that config before running the container.

Example config for **100 Train samples, stratified by illness**:

```json
{
  "data_config_path": "Config/dataloader_config.json",
  "data_mode": "default",
  "data_config_override": {
    "dataset_root": "Data/extracted_data",
    "split": "Train",
    "percent": 100,
    "sampling": "stratified",
    "seed": 42,
    "max_samples": 100,
    "modes": {}
  },
  "whisper_model": "v2",
  "output_dir": "Data/embeddings/whisper_v2_embeddings",
  "mapping_path": "Data/embeddings/whisper_v2_embeddings/mapping.json",
  "overwrite": false
}
```

Then run:

```bash
docker build -f docker/embedding/Dockerfile -t embedding-extractor .
docker run --rm -v %cd%:/app embedding-extractor
```

## Sampling commands for embedding extraction

These commands generate a temporary config and then run the extractor with it.
They will create embeddings and write a `mapping.json`.

### 1) Extract embeddings for 100% of the Train data (random)

```bash
python -c "import json, tempfile; cfg={'data_config_path':'Config/dataloader_config.json','data_mode':'default','data_config_override':{'dataset_root':'Data/extracted_data','split':'Train','percent':100,'sampling':'random','seed':42,'max_samples':None,'modes':{}},'whisper_model':'v2','output_dir':'Data/embeddings/whisper_v2_embeddings','mapping_path':'Data/embeddings/whisper_v2_embeddings/mapping.json','overwrite':False}; f=tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False); json.dump(cfg,f); f.close(); from Data.embedding_extraction import extract_embeddings; print(extract_embeddings(f.name))"
```

### 2) Extract embeddings for exactly 100 Train samples (random)

```bash
python -c "import json, tempfile; cfg={'data_config_path':'Config/dataloader_config.json','data_mode':'default','data_config_override':{'dataset_root':'Data/extracted_data','split':'Train','percent':100,'sampling':'random','seed':42,'max_samples':100,'modes':{}},'whisper_model':'v2','output_dir':'Data/embeddings/whisper_v2_embeddings','mapping_path':'Data/embeddings/whisper_v2_embeddings/mapping.json','overwrite':False}; f=tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False); json.dump(cfg,f); f.close(); from Data.embedding_extraction import extract_embeddings; print(extract_embeddings(f.name))"
```

### 3) Extract embeddings for 100 Train samples, stratified by illness

```bash
python -c "import json, tempfile; cfg={'data_config_path':'Config/dataloader_config.json','data_mode':'default','data_config_override':{'dataset_root':'Data/extracted_data','split':'Train','percent':100,'sampling':'stratified','seed':42,'max_samples':100,'modes':{}},'whisper_model':'v2','output_dir':'Data/embeddings/whisper_v2_embeddings','mapping_path':'Data/embeddings/whisper_v2_embeddings/mapping.json','overwrite':False}; f=tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False); json.dump(cfg,f); f.close(); from Data.embedding_extraction import extract_embeddings; print(extract_embeddings(f.name))"
```

## Short note on data extraction

`Data/data_extraction.py` extracts the nested `.tar` archives from
`Data/raw_data/Downsampled` into `Data/extracted_data`, preserving the `Dev/`
and `Train/` folder structure.
