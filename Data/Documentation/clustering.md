# Clustering Embeddings

## What `clustering.py` does

`Data/clustering.py` loads `.npy` embeddings from a directory, clusters them
using the selected algorithm (HDBSCAN or Spectral), and writes both hard and
soft cluster assignments to JSON files in the output directory.

Each JSON entry includes the embedding ID (derived from the `.npy` filename)
so you can map clusters back to the original `.wav` file.

## Why reduce dimensions before HDBSCAN?

HDBSCAN (and other distance-based clustering algorithms) can struggle as
dimensions grow because distances become less informative (distance
concentration). This is a well-known effect in high-dimensional spaces and is
one of the reasons many workflows reduce embeddings (e.g., PCA or UMAP) before
clustering.

If you want references:
- Aggarwal, Hinneburg, Keim (2001) "On the Surprising Behavior of Distance Metrics in High Dimensional Space"
- Beyer et al. (1999) "When Is 'Nearest Neighbor' Meaningful?"
- McInnes, Healy, Astels (2017) "HDBSCAN: Hierarchical Density Based Clustering"
- McInnes, Healy, Melville (2018) "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction"

## Suggested pipeline

1. Pool frame-level embeddings into one vector per file (default is mean).
2. Optionally reduce to 50-100 dims with PCA or UMAP.
3. Cluster with HDBSCAN.
4. Visualize clusters in 2D using UMAP (or PCA).

## Notes from current runs

- Dimensionality reduction (e.g., `--reduce pca --reduce-dim 50`) produced more distinct groupings in the plot, which is expected for high-dimensional embeddings.
- If you run with `--metric cosine`, HDBSCAN cannot produce prediction data for soft memberships. In that case the script falls back to hard one-hot probabilities, so the soft JSON will contain 0.0/1.0 values.
- For real soft probabilities, use `--metric euclidean` (or omit `--metric`) so HDBSCAN can generate prediction data.

python -m Data.clustering --reduce pca --reduce-dim 50 --plot umap --min-cluster-size 5 --min-samples 5 --metric cosine

## Commands

### Default (HDBSCAN on Whisper v2 embeddings)

```bash
python -m Data.clustering
```

### Use Spectral clustering

```bash
python -m Data.clustering --algorithm spectral
```

### Reduce dimensions before clustering

```bash
python -m Data.clustering --reduce pca --reduce-dim 50
```

### Create a 2D plot of clusters (UMAP)

```bash
python -m Data.clustering --plot umap
```

### Pool frame-level embeddings by flattening

```bash
python -m Data.clustering --pooling flatten
```

### Custom embedding/output directories

```bash
python -m Data.clustering --embedding-dir Data/embeddings/whisper_v2_embeddings --output-dir Data/embeddings/whisper_v2_embeddings_clustered
```
