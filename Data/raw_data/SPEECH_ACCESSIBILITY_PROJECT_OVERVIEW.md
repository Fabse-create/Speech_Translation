# Speech Accessibility Project - Dataset Overview

## Project Background

The **Speech Accessibility Project** is led by the **University of Illinois Urbana-Champaign** and is supported by major technology companies including:
- Amazon
- Apple
- Google
- Meta
- Microsoft

### Mission
To improve Automatic Speech Recognition (ASR) systems for individuals with speech disabilities, making technology more accessible to people with diverse speech patterns.

---

## Dataset Information

### Your Data Collection (Downloaded: 2025-11-02)

You have the **Downsampled** version of the Speech Accessibility Project dataset organized into:

#### **Training Set**
- **833 participants** (JSON metadata files)
- **4 audio archive files** (000, 001, 002, 003)
- 1 consolidated JSON metadata archive

#### **Development/Validation Set**
- **126 participants** (JSON metadata files)
- **1 audio archive file** (000)
- 1 consolidated JSON metadata archive

### File Structure

```
data/raw/Downsampled/
├── Train/
│   ├── SpeechAccessibility_2025-11-02_000.tar  # Audio files (subset 1)
│   ├── SpeechAccessibility_2025-11-02_001.tar  # Audio files (subset 2)
│   ├── SpeechAccessibility_2025-11-02_002.tar  # Audio files (subset 3)
│   ├── SpeechAccessibility_2025-11-02_003.tar  # Audio files (subset 4)
│   └── SpeechAccessibility_2025-11-02_Train_Only_Json.tar  # Metadata
└── Dev/
    ├── SpeechAccessibility_2025-11-02_000.tar  # Audio files
    └── SpeechAccessibility_2025-11-02_Dev_Only_Json.tar  # Metadata
```

---

## Data Organization

### Audio Archive Structure

Each participant's audio data is stored in a tar archive containing:
```
{participant-uuid}.tar
├── {participant-uuid}/
│   ├── {participant-uuid}.json  # Metadata for this participant
│   ├── {participant-uuid}_101_1169_16kHz.wav  # Audio file 1
│   ├── {participant-uuid}_1052_1165_16kHz.wav  # Audio file 2
│   └── ...  # Multiple audio recordings
```

### Audio File Naming Convention

Format: `{participant-uuid}_{number1}_{number2}_16kHz.wav`
- **Sampling Rate**: 16 kHz (downsampled for efficient processing)
- **Format**: WAV (uncompressed audio)

---

## Metadata Structure

Each participant has a JSON file with comprehensive information:

### Participant Information
```json
{
  "Contributor ID": "f8e65210-0ac9-4c7c-c658-08db73ef9d1c",
  "Do you consider U.S. English to be your first language": "Yes",
  "What language(s) do you speak": "No Response Supplied",
  "What age did you start speaking English": "No Response Supplied",
  "I consent to have my voice samples shared...": "Yes",
  "Etiology": "Parkinson's Disease",
  "BlockNumber": 5
}
```

### Speech Disabilities Covered

The dataset includes participants with:
- **Parkinson's Disease**
- **Down Syndrome**
- **ALS (Amyotrophic Lateral Sclerosis)**
- **Cerebral Palsy**
- **Stroke survivors**
- Other speech-affecting conditions

### File Entries

Each audio recording has detailed metadata:

```json
{
  "Filename": "f8e65210-0ac9-4c7c-c658-08db73ef9d1c_1133_1447_16kHz.wav",
  "Created": "2023-07-13 18:15:53",
  "CreatedOrModified": "2023-07-16 15:46:08",
  "Comment": "",
  "Prompt": {
    "Prompt Text": "Make it cooler.",
    "Transcript": "Make it cooler.",
    "Category Description": "Digital Assistant Commands",
    "Sub Category Description": "Control Your Smart Home"
  },
  "Ratings": [
    {
      "Level": "1",
      "Comment": "",
      "Dimension Description": "Breathy voice (continuous)",
      "Dimension Category Description": "Parkinson Disease",
      "RatingDate": "2023-10-03 09:08:11"
    },
    {
      "Level": "1",
      "Dimension Description": "Imprecise consonants",
      "Dimension Category Description": "Parkinson Disease",
      "RatingDate": "2023-10-03 09:08:11"
    },
    {
      "Level": "1",
      "Dimension Description": "Inappropriate silences",
      "Dimension Category Description": "Parkinson Disease",
      "RatingDate": "2023-10-03 09:08:11"
    },
    {
      "Level": "1",
      "Dimension Description": "Intelligibility",
      "Dimension Category Description": "Parkinson Disease",
      "RatingDate": "2023-10-03 09:08:11"
    }
  ]
}
```

---

## Speech Prompt Categories

Based on the sample data, the dataset includes various categories:

### 1. **Digital Assistant Commands**
   - **Control Your Smart Home**: "Make it cooler", "Turn up the temperature 1 degree"
   - Everyday voice commands for smart devices

### 2. **Novel Sentences**
   - Complete sentences with varied vocabulary
   - Example: "A vague fear crept over her that he might finally succeed in proving to her that it was her duty to resign."

### 3. **Other Categories** (to be discovered when extracting full data)
   - Likely includes: Common phrases, numbers, dates, navigation commands, etc.

---

## Speech Characteristics Annotations

The dataset includes detailed **ratings** for speech characteristics, particularly for Parkinson's Disease:

### Rating Dimensions
1. **Breathy voice (continuous)** - Continuous airflow during speech
2. **Imprecise consonants** - Difficulty articulating consonant sounds
3. **Inappropriate silences** - Unexpected pauses in speech
4. **Intelligibility** - Overall understandability of speech
5. And potentially more dimensions for other conditions

### Rating Scale
- Levels typically range from 1-5 or 1-7
- **Level 1**: Minimal/mild impairment
- **Higher levels**: More severe impairment

---

## Key Features of This Dataset

### 1. **Real-World Speech Patterns**
- Authentic recordings from people with speech disabilities
- Natural variations in speech characteristics
- Diverse severity levels

### 2. **Rich Annotations**
- Original prompts provided
- Human transcriptions included
- Professional speech pathologist ratings
- Specific characteristic dimensions annotated

### 3. **Ethical Data Collection**
- Explicit participant consent
- Privacy protection through UUIDs
- Research-focused usage agreements

### 4. **Multiple Use Cases**
- ASR system training and fine-tuning
- Speech disability classification
- Assistive technology development
- Clinical research on speech patterns

---

## Comparison: Speech Accessibility Project vs TORGO

Your project originally used **TORGO**, but the Speech Accessibility Project offers several advantages:

| Feature | TORGO | Speech Accessibility Project |
|---------|-------|------------------------------|
| **Size** | ~21 hours (15 speakers) | **Much larger** (959 participants in your dataset) |
| **Participants** | 8 dysarthric, 7 controls | **833 train + 126 dev** participants |
| **Conditions** | CP, ALS only | **Parkinson's, Down Syndrome, ALS, CP, Stroke, etc.** |
| **Annotations** | Basic transcriptions | **Detailed speech characteristic ratings** |
| **Data Quality** | Older dataset (2012) | **Recent** (2023-2025 recordings) |
| **Variety** | Limited prompt types | **Multiple categories** (commands, sentences, etc.) |
| **Support** | Academic project | **Industry-backed** (Amazon, Apple, Google, Meta, Microsoft) |

---

## Data Usage Recommendations

### For Your MoE-Whisper Project

#### 1. **Initial Exploration** (Recommended First Steps)
   - Extract Dev set (smaller, 126 participants)
   - Analyze distribution of speech disabilities
   - Examine audio quality and duration statistics
   - Review transcription accuracy

#### 2. **Preprocessing Pipeline** (Adapt existing code)
   - Modify `datapreprocessing/prepare_raw_dataset.py` to:
     - Parse JSON metadata structure
     - Extract audio files from nested tar archives
     - Create labels.jsonl in the same format as TORGO
     - Preserve speech characteristic ratings for potential use

#### 3. **Clustering Strategy**
   - Can apply same clustering methods (BGMM, Spectral, HDBSCAN+UMAP)
   - Consider clustering by:
     - **Etiology** (Parkinson's, ALS, etc.)
     - **Acoustic features** (as in original approach)
     - **Speech characteristic ratings** (new possibility!)
     - **Intelligibility levels**

#### 4. **Expert Specialization Opportunities**
   - **Expert 1**: Parkinson's Disease speech patterns
   - **Expert 2**: Down Syndrome characteristics
   - **Expert 3**: ALS/Stroke patterns
   - **Expert 4**: Cerebral Palsy or mixed conditions

   OR use unsupervised clustering as before

#### 5. **Evaluation Advantages**
   - Much larger test set (126 participants in Dev)
   - Can stratify evaluation by:
     - Disability type
     - Severity level
     - Intelligibility ratings
     - Prompt category

---

## Next Steps

### Immediate Actions

1. **Extract and Explore Dev Set**
   ```bash
   # Create extraction script for the tar archives
   # Parse JSON metadata to understand data distribution
   ```

2. **Create Data Statistics**
   - Total hours of audio
   - Distribution by etiology
   - Number of recordings per participant
   - Prompt category breakdown

3. **Adapt Preprocessing Scripts**
   - Modify `prepare_raw_dataset.py` for new data structure
   - Update paths in `config/paths.yaml`
   - Test with small subset first

4. **Quality Assessment**
   - Check audio quality (sample rate, noise levels)
   - Validate transcription accuracy
   - Verify rating annotations

### Research Opportunities

1. **Condition-Specific Experts**: Train experts specialized for each disability type
2. **Severity-Aware Routing**: Gate network considers severity ratings
3. **Multi-Task Learning**: Predict both transcription + speech characteristics
4. **Transfer Learning**: Pre-train on this larger dataset, fine-tune on TORGO
5. **Prompt-Category Adaptation**: Specialize experts by command types

---

## Data Ethics & Usage

### Important Considerations

1. **Privacy**: All participants are de-identified with UUIDs
2. **Consent**: Participants consented to research use
3. **Purpose**: Data intended for improving accessibility technology
4. **Attribution**: Acknowledge the Speech Accessibility Project in publications

### Citation

When using this data, cite:
```
Speech Accessibility Project
University of Illinois Urbana-Champaign
Supported by: Amazon, Apple, Google, Meta, Microsoft
https://speechaccessibilityproject.beckman.illinois.edu/
```

---

## Additional Resources

- **Official Website**: https://speechaccessibilityproject.beckman.illinois.edu/
- **Data Access FAQ**: https://speechaccessibilityproject.beckman.illinois.edu/conduct-research-through-the-project/faq-on-using-our-data
- **Transcription Guidelines**: See `data/raw/Transcription_Guidelines.pdf` in your project
- **Speech Prompts**: See `data/raw/SpeechPrompts.docx` in your project

---

## Summary

You now have access to a **significantly larger and more diverse** dataset than TORGO, with:
- ✅ **959 participants** (vs 15 in TORGO)
- ✅ **Multiple disability types** (vs just CP and ALS)
- ✅ **Rich annotations** (speech characteristic ratings)
- ✅ **Recent recordings** (2023-2025)
- ✅ **Industry support** and ongoing updates

This dataset offers excellent opportunities to:
1. Train more robust speech recognition models
2. Develop condition-specific experts in your MoE architecture
3. Achieve better generalization across speech disabilities
4. Potentially publish impactful research on accessible ASR

**Your MoE-Whisper project is well-positioned to leverage this dataset for significant improvements in dysarthric/atypical speech recognition!**

---

## Detailed Dataset Statistics (Dev Set Analysis)

### Scale and Coverage
- **Total Participants**: 126
- **Total Recordings**: 47,836 audio files  
- **Average Recordings per Participant**: ~380 recordings
- **Recording Range**: 2 to 450 recordings per participant

### Language and Consent
- **69.0%** (87) consider U.S. English as first language
- **88.9%** (112) consented to public sharing at research conferences

### Etiology Breakdown (Dev Set)

| Condition | Count | Percentage |
|-----------|-------|------------|
| **Parkinson's Disease** | 44 | 34.9% |
| **ALS** | 36 | 28.6% |
| **Cerebral Palsy** | 25 | 19.8% |
| **Down Syndrome** | 14 | 11.1% |
| **Stroke** | 7 | 5.6% |

### Prompt Category Distribution

| Category | Recordings | Description |
|----------|------------|-------------|
| **Digital Assistant Commands** | 34,655 (72.5%) | Voice commands for smart devices |
| **Novel Sentences** | 6,411 (13.4%) | Complete sentences with varied vocabulary |
| **Spontaneous Speech Prompts** | 5,903 (12.3%) | Open-ended questions and responses |
| **Non-spontaneous Speech Prompts** | 867 (1.8%) | Structured reading tasks |

### Detailed Subcategories (Top 15)

1. **Playing and Controlling Media**: 9,041 recordings
2. **Everyday Tasks and Planning**: 5,694 recordings
3. **Open-ended questions**: 3,363 recordings
4. **Get News, Weather, and Traffic Updates**: 3,233 recordings
5. **Control Your Smart Home**: 2,833 recordings
6. **Online Search**: 2,765 recordings
7. **Phone Operations**: 2,391 recordings
8. **Wake Phrases**: 1,962 recordings
9. **Procedural questions**: 954 recordings
10. **Plan a Trip or an Outdoor Visit**: 907 recordings
11. **Open-ended pair question 1**: 805 recordings
12. **Open-ended pair question 2**: 781 recordings
13. **Broadcast over more than one speaker**: 758 recordings
14. **5k**: 450 recordings (likely 5000 most common words)
15. **UA-uncommon**: 224 recordings (Uncommon words/utterances)

### Speech Characteristic Ratings (Expert Annotations)

The dataset includes detailed perceptual ratings by speech-language pathologists on multiple dimensions:

#### Most Common Rating Dimensions:

1. **Monoloudness**: 1,389 annotations (consistent loudness issues)
2. **Imprecise consonants**: 1,388 annotations
3. **Reduced stress**: 1,379 annotations  
4. **Monopitch**: 1,378 annotations (lack of pitch variation)
5. **Harsh voice**: 1,184 annotations
6. **Naturalness**: 1,177 annotations
7. **Inappropriate silences**: 1,162 annotations
8. **Pitch level**: 974 annotations
9. **Breathy voice (continuous)**: 954 annotations
10. **Intelligibility**: 954 annotations

#### Additional Rating Dimensions:

- **Voice Quality**: Strained-strangled voice, hoarse/wet voice, voice tremor
- **Prosody**: Atypical prosody, excess/equal stress, short phrases
- **Resonance**: Hypernasality, hyponasality, atypical resonance
- **Articulation**: Distorted vowels, prolonged phonemes, irregular articulatory breakdowns
- **Rate/Timing**: Variable rate, slow rate, short rushes of speech, prolonged intervals
- **Respiration**: Audible inspiration, forced inspiration-expiration
- **Other**: Repeated phonemes, voice stoppages, nasal emission

---

## Transcription Guidelines Summary

The dataset follows rigorous **verbatim transcription** standards with intelligent handling of speech characteristics:

### Key Transcription Principles

#### 1. **Verbatim with Special Markup**
   - All words transcribed exactly as spoken
   - **Disfluencies preserved** (false starts, repetitions, fillers)
   - Special codes for various speech phenomena

#### 2. **Disfluency Handling**

| Type | Notation | Example |
|------|----------|---------|
| **False starts** | `(partial)` | `(It's in the) It takes place in the future` |
| **Full-word repetitions** | `(word)` | `(And) And we would go on trips` |
| **Partial words** | `(s-)` | `My favorite (s-) sandwich` |
| **Stuttering** | `(f-*)` | `(f-*) Find m-my phone` |
| **Filler words** | `(uh)` | `And (uh) it's a dry humor` |
| **Sound prolongation** | Not marked | - |

#### 3. **Special Markings**

| Marking | Purpose | Example |
|---------|---------|---------|
| `~` | Individual spoken letters | `~J ~E ~S ~S ~Y ~E` |
| `~` | Acronyms (letter-by-letter) | `~NPR station` |
| `[PII]` | Personal Identifiable Information | `together with [PII], our son` |
| `{g: word}` | Best guess (uncertain) | `she was my {g: idol}` |
| `{w: 3}` | Unknown words (count known) | `was {w: 3} in the article` |
| `{u: }` | Completely unintelligible | `Replace {u: } very tightly` |
| `(cs:)` | Caregiver speech (Parkinson's) | `(cs: Go ahead.)` |
| `(ss:)` | Spontaneous speech in reading task | `Call coffee shop (ss: Hope not Starbucks)` |
| `{f:}` | Foreign language sentence | `{f:}` (no transcription) |
| `[ ]` | Prompt text or comments | `[Talk about childhood memory]` |

#### 4. **Privacy Protection**
   - Participant names → `[PII]`
   - Family member names → `[PII]`
   - Addresses, phone numbers, emails → `[PII]`
   - Audio sections with PII are **silenced** in the files

#### 5. **Numbers and Formatting**
   - Numbers written out: `twenty-three` (not `23`)
   - Contractions preserved: `I'm`, `gonna`, `You're`
   - Proper capitalization and punctuation
   - Interjections transcribed: `yeah`, `whoa`, `huh`, `okay`

#### 6. **Special Cases for Different Etiologies**

- **Cognitive/Reading Challenges** (e.g., aphasia from stroke):
  - False starts NOT marked with parentheses
  - Example: `If animals could talk, can where did do you like you like to say?`

- **Caregiver Assistance**:
  - For Parkinson's: marked as `(cs: text)`
  - For others: Either silenced or time-stamped transcription method

#### 7. **Invalid Audio Codes**

| Code | Meaning |
|------|---------|
| `nod` | No data (file doesn't exist) |
| `nov` | No voice or irrelevant voice |
| `bgn` | Background noise masks voice |
| `otv` | Other voice(s), not participant |
| `cut` | Voice cut off (less than one word) |
| `dis` | Severely distorted/unintelligible sound |
| `loa` | Low amplitude (<5dB) |
| `nrw` | No real words in English (aphasia) |

---

## Full Prompt List Resource

According to the `SpeechPrompts.docx` file in your dataset:

**Complete prompt lists are available at**:  
https://github.com/speechaccessibility/PromptsAnnotationGuidelines

This repository contains:
- Full list of digital assistant commands
- Novel sentence prompts
- Spontaneous speech questions
- Annotation guidelines for rating speech characteristics

---

## Implications for Your MoE-Whisper Project

### 1. **Data Quality Advantages**

✅ **Professional Transcriptions**: All transcripts follow rigorous guidelines with SLP oversight  
✅ **Preserved Speech Patterns**: Disfluencies, repetitions, and partial words are marked systematically  
✅ **Expert Annotations**: Detailed perceptual ratings on ~40+ speech dimensions  
✅ **Clean Handling of Edge Cases**: Privacy protection, caregiver speech, unintelligible sections

### 2. **Model Training Opportunities**

#### **Multi-Task Learning**
- **Primary task**: ASR (speech-to-text)
- **Auxiliary tasks**:
  - Predict disfluency types
  - Estimate speech characteristic severity
  - Classify etiology from acoustic patterns
  - Predict intelligibility ratings

#### **Expert Specialization Strategies**

**Option A: Etiology-Based Experts**
```
Expert 1: Parkinson's Disease (44 participants, 34.9%)
Expert 2: ALS (36 participants, 28.6%)
Expert 3: Cerebral Palsy (25 participants, 19.8%)
Expert 4: Down Syndrome + Stroke (21 participants, 16.7%)
```

**Option B: Characteristic-Based Experts**
```
Expert 1: Monopitch/Monoloudness patterns
Expert 2: Imprecise consonants/articulation issues
Expert 3: Timing issues (inappropriate silences, variable rate)
Expert 4: Voice quality (harsh, breathy, strained)
```

**Option C: Prompt-Type Experts**
```
Expert 1: Digital assistant commands (72.5% of data)
Expert 2: Novel sentences (13.4%)
Expert 3: Spontaneous speech (12.3%)
Expert 4: Mixed/other
```

#### **Gating Network Design**

The gating network could be conditioned on:
- **Metadata features**: Etiology, prompt category, speaker ID
- **Acoustic features**: Pitch variation, speaking rate, spectral characteristics
- **Predicted characteristics**: Intelligibility estimate, severity level
- **Hybrid approach**: Combine acoustic and metadata signals

### 3. **Preprocessing Adaptations**

#### **Handling Special Tokens**

Your preprocessing pipeline should:
1. **Extract clean text** for WER calculation:
   - Remove markup: `(uh)`, `{g: word}`, `[PII]`, etc.
   - Keep actual spoken words only

2. **Optionally preserve disfluency information**:
   - Create parallel labels for disfluency detection
   - Could improve model's handling of atypical speech patterns

3. **Handle intelligibility markers**:
   - Skip or specially handle `{u: }` sections
   - Weight `{g: word}` sections lower in loss calculation

#### **Data Stratification**

For robust evaluation:
- **By etiology**: Ensure all conditions represented in train/dev/test
- **By severity**: Balance across intelligibility ratings
- **By prompt type**: Test generalization across command types
- **By recording quality**: Include diverse audio conditions

### 4. **Evaluation Metrics**

Beyond standard WER:
- **Stratified WER** by:
  - Etiology type
  - Intelligibility rating (from annotations)
  - Prompt category
  - Speech characteristic presence
- **Character Error Rate (CER)**: Important for partial-word handling
- **Disfluency-aware metrics**: Separate WER for fluent vs. disfluent segments
- **Clinical relevance**: Correlation with SLP intelligibility ratings

### 5. **Expected Improvements Over TORGO**

| Aspect | TORGO | Speech Accessibility Project |
|--------|-------|------------------------------|
| **Training data** | ~7.3 hrs dysarthric | **~380 recordings × 833 = ~300,000+ recordings** |
| **Diversity** | 2 conditions (CP, ALS) | **5 conditions + varied severity** |
| **Annotations** | Basic | **40+ perceptual dimensions rated** |
| **Prompt variety** | Limited | **4 major categories, 30+ subcategories** |
| **Transcription** | Simple | **Detailed verbatim with markup** |
| **Generalization** | Narrow | **Much broader population coverage** |

### 6. **Research Publication Potential**

With this dataset, you can investigate:
1. **Novel research questions**:
   - Do condition-specific experts outperform acoustic clustering?
   - Can models learn to route based on speech characteristics?
   - Multi-task learning: ASR + characteristic prediction
   - Transfer learning from large dysarthric corpus to TORGO

2. **Stronger baselines**:
   - Much larger scale than previous dysarthric ASR work
   - Industry-backed dataset with ongoing updates
   - Reproducible results with clear data split

3. **Clinical applications**:
   - Correlation between model confidence and intelligibility ratings
   - Severity-aware ASR systems
   - Assistive technology deployment insights

---

## Next Steps for Implementation

### Phase 1: Data Extraction & Exploration (Week 1)

1. **Extract full datasets**:
   ```bash
   python -m datapreprocessing.extract_sap_data --split Dev --extract_samples --num_samples 10
   python -m datapreprocessing.extract_sap_data --split Train
   ```

2. **Analyze audio characteristics**:
   - Duration distribution
   - Sample rate verification (16kHz)
   - Audio quality assessment
   - Silence/noise analysis

3. **Parse transcriptions**:
   - Create clean text versions (remove markup)
   - Extract disfluency annotations
   - Build etiology-speaker mapping
   - Extract speech characteristic ratings

### Phase 2: Preprocessing Pipeline (Week 2)

1. **Adapt `prepare_raw_dataset.py`**:
   - Parse nested tar structure
   - Extract JSON metadata
   - Generate `labels.jsonl` compatible with existing pipeline
   - Create train/dev/test splits (use provided Dev set as test)

2. **Extract Whisper embeddings**:
   - Run encoder on all audio files
   - Save embeddings with metadata
   - Link to speaker ID and etiology

3. **Generate cluster assignments**:
   - Option A: Use etiology labels directly
   - Option B: Apply HDBSCAN/BGMM/Spectral on embeddings
   - Option C: Cluster by speech characteristics

### Phase 3: Model Training (Week 3-4)

1. **Stage 1: Expert Pre-training**:
   - Train LoRA experts on each cluster
   - Monitor convergence per expert
   - Save expert checkpoints

2. **Stage 2: Joint MoE Training**:
   - Initialize with pre-trained experts
   - Train gating network
   - Apply load balancing losses
   - Temperature annealing

3. **Evaluation**:
   - Overall WER on Dev set
   - Stratified WER by etiology
   - Stratified WER by intelligibility
   - Expert utilization analysis

### Phase 4: Analysis & Iteration (Week 5)

1. **Error analysis**:
   - Which etiologies have highest WER?
   - Do experts specialize as expected?
   - Correlation with speech characteristic ratings

2. **Hyperparameter tuning**:
   - Number of experts
   - Top-k routing
   - LoRA rank
   - Auxiliary loss weights

3. **Ablation studies**:
   - Compare to baseline Whisper
   - Compare to single LoRA adaptation
   - Compare clustering strategies

---

## File Locations Summary

All extracted analysis files are located in `data/sap_analysis/`:

- `dev_statistics.json` - Quantitative statistics for Dev set
- `SpeechPrompts_extracted.txt` - Link to GitHub prompt repository
- `Transcription_Guidelines_extracted.txt` - Phase 1 transcription rules
- `Transcription_Guidelines_Phase2_extracted.txt` - Phase 2 rules (all etiologies)

Original documentation files in `data/raw/`:
- `SpeechPrompts.docx` - Prompt list reference
- `Transcription_Guidelines.pdf` - Phase 1 rules (Parkinson's)
- `Transcription Guidelines Phase2.pdf` - Phase 2 rules (all etiologies)
- `SPEECH_ACCESSIBILITY_PROJECT_OVERVIEW.md` - This comprehensive overview

---

## Conclusion

You now have access to a **world-class dataset** for dysarthric speech recognition with:

✅ **959 total participants** (833 train + 126 dev)  
✅ **47,836+ recordings** in Dev set alone (likely 300,000+ total)  
✅ **5 different etiologies** with varied severity levels  
✅ **Professional transcriptions** with detailed markup  
✅ **40+ speech characteristic dimensions** annotated by SLPs  
✅ **4 major prompt categories** for diverse evaluation  
✅ **Industry backing** from top tech companies  
✅ **Rigorous privacy protection** and ethical standards

This dataset positions your MoE-Whisper project to:
1. **Significantly outperform** previous dysarthric ASR systems
2. **Publish impactful research** on accessible speech technology
3. **Develop clinically-relevant** ASR systems
4. **Advance the field** of assistive technology

**Your MoE-Whisper project is well-positioned to leverage this dataset for significant improvements in dysarthric/atypical speech recognition!**
