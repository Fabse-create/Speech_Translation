# Clustering Embeddings

## What `clustering.py` does

`Data/clustering.py` loads `.npy` embeddings from a directory, clusters them
using the selected algorithm (HDBSCAN or Spectral), and writes both hard and
soft cluster assignments to JSON files in the output directory.

Each JSON entry includes the embedding ID (derived from the `.npy` filename)
so you can map clusters back to the original `.wav` file.

## Commands

### Default (HDBSCAN on Whisper v2 embeddings)

```bash
python Data/clustering.py
```

If you prefer a containerized run, use Docker:

```bash
docker build -f docker/clustering/Dockerfile -t embedding-clustering .
docker run --rm -v %cd%:/app embedding-clustering
```

### Use Spectral clustering

```bash
python Data/clustering.py --algorithm spectral
```

### Custom embedding/output directories

```bash
python Data/clustering.py --embedding-dir Data/embeddings/whisper_v2_embeddings --output-dir Data/embeddings/whisper_v2_embeddings_clustered
```
