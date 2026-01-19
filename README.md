# Speech Translation – Mixture of Experts (MoE) with Whisper because Stefan is the best

This project explores **speech-to-text translation (EN → DE)** using a **Mixture-of-Experts (MoE)** architecture built on **Whisper v2**, combined with **DeepL** for machine translation and **SALMONN** as an end-to-end baseline.

---

## ✅ TODO / Status

* ~~Mail (Trennung, neuer Plan, Erwartungen)~~
* Struktur (Stefan)
* Translation Script (Fabian) mit google und dann OPUS finetunen
* ~~SALMONN (Fabian)~~
* Data preprocessing for Whisper
* Embedding extraction from Whisper v2 (save embeddings)
* HDBSCAN clustering on embeddings (save clusters)
* Pre-train **Gating Model** (save parameters, reuse embeddings)
* Pre-train **Experts** using static G (save parameters, reuse embeddings)
* Train complete ASR (MoE)

---

## 🔁 Translation

**Text translation pipeline (EN → DE):**

* **DeepL**
* Used as a cascaded MT system after ASR
* Later: fine-tuned DeepL variant integrated into training

---

## 🧱 Architecture / Structure

### Base Model

* **Whisper v2 (medium)**

  * Encoder → Latent Space → Decoder
  * Encoder embedding dimension: **1280**

### Data Flow (Preprocessing & Clustering)

```
Audio Data (≈20% sampled)
   │
   ▼
Whisper v2 Encoder
   │
   ▼
Latent Space / Embeddings (E ∈ R¹²⁸⁰)
   │
   ▼
HDBSCAN Clustering
   │
   ▼
Cluster Labels (y)
```

The cluster labels **y** are used to pre-train the **Gating Model**.

---

## 🧠 Gating Model (G)

The gating network assigns embeddings to experts.

**Architecture:**

```
Linear(1280 → 512)
ReLU
Linear(512 → 10)   # number of experts
```

Output is passed through **softmax** to obtain expert weights.

---

## 🚀 Training Plan

### 1️⃣ Gating Model Pre-Training

**Objective:** Learn cluster-aware routing.

```
Input:  Embeddings E
Target: Cluster labels y

Loss:   Cross-Entropy
Train:  G(E) → y
```

Embeddings are reused (no re-encoding).

---

### 2️⃣ Expert Pre-Training (Static G)

**Data:** ~20% sampled dataset

```
Audio x, Transcription yᵢ
   │
   ▼
Whisper v2 Encoder
   │
   ▼
Embeddings E
   │
   ▼
Softmax(G(E))  → expert selection
   │
   ▼
Whisper v2 Decoder (Expert-specific)
```

* **LoRA fine-tuning** per expert
* **Local loss only**
* Gating model frozen

---

### 3️⃣ Final End-to-End Training (MoE)

**Data split:**

* 80% train
* 10% validation
* 10% evaluation

```
Audio x, Transcription yᵢ
   │
   ▼
Whisper v2 Encoder
   │
   ▼
Embeddings E
   │
   ▼
Softmax(G(E))  → expert distribution
   │
   ▼
Whisper v2 Decoder (LoRA Experts)
```

* **Global loss** (with auxiliary routing loss)
* Gradients flow back into:

  * Experts
  * Gating Model

---

### 4️⃣ ASR + MT Integration

* Add **DeepL** to ASR output
* Retrain / fine-tune pipeline
* Optional: fine-tuned DeepL variant

---

## 📊 Evaluation

### Baselines & Comparisons

**ASR + MT Cascaded:**

* Whisper v2 (no fine-tuning) + DeepL
* Whisper v3 (no fine-tuning) + DeepL
* Whisper v2 + LoRA fine-tuning + DeepL

**MoE Models:**

* Whisper v2 MoE + DeepL
* Whisper v2 MoE + fine-tuned DeepL

**End-to-End Models:**

* SALMONN (no fine-tuning)
* SALMONN (end-to-end fine-tuned)

**MT Only:**

* DeepL
* Fine-tuned DeepL

---

### Metrics

* **WER** (Word Error Rate)
* **BLEU** (Translation Quality)

---

### Error Analysis

* Where do errors occur?
* Performance grouped by **disease / impairment type**
* Identification of systematic weaknesses

---

## ⚠️ Known Weaknesses / Risks

* Limited **healthy audio** baseline
* Noisy or low-quality recordings
* **Nationality bias** (~80% USA speakers)
* Potential cluster imbalance
