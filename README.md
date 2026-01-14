# Speech_Translation

## TODO
- Mail (Trennung, neuer Plan, Was erwartet sie?)
- Datapreprocessing for Whisper 
- Embedding extraxtion from Whisper v2; save Embeddings
- HDBSCAN with embeddings; save Clustering
- Pre train Gatingmodel; sava Parameters; use Embedding extraxtion from before
- Pre train Experts using static G with saved parameters; save parameters for Experts; use Embedding extraxtion from before
- Train complete ASR

## Translation

Pipeline für Übersetzung der Texte von Englisch nach Deutsch
- DEEPL

## Struktur

Whisper v2: Encoder LatentSpace Decoder
Daten (circa 20% gesampled)<br>
  |<br>
Whisper v2 Encoder<br>
  |<br>
LatentSpace / Embeddings(E) durch whisperv2 medium eine Dimensionalität von 1280<br>
  |<br>
Clustering HDBSCAN<br>
  |<br>
Labels(y) für Daten zum vor trainieren des Gating Model

Gating Model(G):<br>
lin(1280, 512)<br>
ReLU(x)<br>
lin(512, 10)

## Plan

1. Gating Model pre training:<br>
G(E, y)

2. Expert pre Training:<br>
Daten (circa 20% gesampled) x, y_i <br>
  | <br>
Whisper v2 Encoder<br>
  |<br>
LatentSpace / Embeddings(E) durch whisperv2 medium eine Dimensionalität von 1280<br>
  |<br>
softmax(G(x)) -> reduziert Verteilung auf maximale Anzahl von Experten<br>
  |<br>
Whisper v2 Decoder<br>
LoRA fine tuning (local loss)


3. Final Training<br>
Daten (circa 80% gesampled, 10% eval, 10% val) x, y_i<br>
  |<br>
Whisper v2 Encoder<br>
  |<br>
LatentSpace / Embeddings(E) durch whisperv2 medium eine Dimensionalität von 1280<br>
  |<br>
softmax(G(x)) -> reduziert Verteilung auf maximale Anzahl von Experten<br>
  |<br>
Whisper v2 Decoder<br>
LoRA fine tuning (gloabl loss probably auxilliary loss, gets backwarded to Gating Model aswell)

4. Add Deepl to ASR and Train again

## Evaluation

- General Performance rated by having an Eval Dataset
  - Whisper v2 without finetuning with DEEPL as MT Cascaded
  - Whisper v3 without finetuning with DEEPL as MT Cascaded
  - Whisper v2 fine tuned with Lora with DEEPL as MT Cascaded
 
  - MoE using Whisper v2 with DEEPL as MT Cascaded
  - MoE using Whisper v2 with DEEPL finetuned as MT Cascaded
 
  - SALMONN without finetuning End to End
  - SALMONN with finetuning End to End
 
 
  - MT
    - DEEPL
    - DEEPL finetuned  
 
- Comparison Values:
  - BLEU
  - WER
 
Wo tretten die Probleme auf?
- Krankheit (sortiert nach welche Krankheit sorgt für was für einen Score)

## Weaknesses
- Healthy Audio
- noisy data?
- nationality bias (80% USA)
- 
