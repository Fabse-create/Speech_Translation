# Etiology Embedding Plot

This script generates a 2D visualization of Whisper embeddings, colored by
illness/impairment (the `Etiology` field in the dataset metadata). It uses the
same dimensionality reduction utilities as the clustering pipeline.

## Usage

```bash
python Clustering_Algorithms/etiology_embedding_plot.py \
  --embedding-dir Data/embeddings/whisper_v2_embeddings \
  --splits Train,Dev \
  --reduce pca \
  --reduce-dim 50 \
  --plot-method umap \
  --output-dir Data/embeddings/whisper_v2_embeddings_etiology
```

## Outputs

The script writes:

- `etiology_plot.png` – a 2D scatter plot colored by etiology
- `etiology_embeddings.csv` – per-sample 2D coordinates with etiology labels

## Notes

- If `mapping.json` exists in the embedding directory, it is used to resolve
  embedding IDs to speaker + filename for accurate etiology lookup.
- If you extracted embeddings from a different split, set `--splits` accordingly.
- For large datasets, consider `--plot-method pca` for faster rendering.
