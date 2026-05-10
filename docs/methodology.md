# Methodology

This document explains the design decisions behind each stage of the analysis. It accompanies the notebooks and is intended as the reference document for reviewers who want to understand *why* things were done a particular way.

---

## 1. Dataset

**Source:** PhysioNet EEG Motor Movement/Imagery Dataset (EEGBCI), Schalk et al. 2004.

**Description:**
- 109 healthy subjects
- 64-channel EEG (international 10-10 system)
- 160 Hz sampling rate
- 14 runs per subject across multiple tasks:
  - Baseline (eyes open / eyes closed)
  - Motor execution (left/right fist, both fists, both feet)
  - Motor imagery (same actions, imagined only)

**Why this dataset:**
- Open-access and well-documented
- Sufficient subjects to support cross-subject analysis
- Contains both execution and imagery — supports ERP comparisons and BCI classification
- Loadable directly via `mne.datasets.eegbci.load_data()` — no manual download

**Subset used in this project:**
Runs 4, 8, 12 (left/right fist motor imagery) for the classification task. All runs for exploratory analysis.

---

## 2. Preprocessing

**Pipeline:**
1. Load raw EDF → MNE Raw object
2. Set standard 10-10 montage
3. Bandpass filter: 1–40 Hz (FIR, zero-phase)
4. Notch filter: 60 Hz (US power line)
5. Re-reference: average reference
6. ICA decomposition (Picard or FastICA, 20 components)
7. Manual or automated artifact component rejection (eye blinks, muscle, line noise)
8. Epoch extraction: -0.5 to 4.0 s around event markers
9. Baseline correction: -0.5 to 0 s
10. Reject epochs exceeding amplitude thresholds (auto-reject)

**Rationale:**
- 1 Hz high-pass removes slow drifts that destabilize ICA
- 40 Hz low-pass keeps mu/beta and lower gamma without aliasing concerns
- Average reference is standard for high-density EEG and avoids reference bias
- 4-second epochs cover the imagery window in the protocol

---

## 3. ERP Analysis

**Goal:** Visualize and compare event-related potentials between motor execution and motor imagery conditions.

**Approach:**
- Average across trials per condition per subject
- Grand-average across subjects
- Statistical comparison via cluster-based permutation tests (non-parametric, multiple-comparisons-safe)
- Topographic maps at peak latencies

**Expected findings:**
- Contralateral negativity over motor cortex (~200–500 ms post-cue)
- Larger amplitude in execution vs. imagery
- Lateralization based on left/right cue

---

## 4. Time-Frequency Analysis

**Goal:** Detect event-related desynchronization (ERD) in the mu (8–13 Hz) and beta (13–30 Hz) bands — the canonical motor imagery signatures.

**Approach:**
- Morlet wavelets, 4–40 Hz, 7-cycle width
- Computed per-trial then averaged
- Baseline correction: log-ratio relative to pre-cue interval
- Statistical thresholding via cluster permutation

**Expected findings:**
- ERD (power decrease) in mu/beta over contralateral motor cortex during imagery
- Beta rebound (ERS) post-movement

---

## 5. Classification

### Classical pipeline: CSP + classifier

**Common Spatial Patterns (CSP):**
- 6 spatial filters (3 per class)
- Log-variance features
- Trained per-subject (within-subject classification)

**Classifiers compared:**
- Linear Discriminant Analysis (LDA) — the BCI literature baseline
- Support Vector Machine with RBF kernel
- Logistic regression as a sanity-check baseline

**Evaluation:**
- 5-fold cross-validation per subject
- Report mean ± std accuracy across subjects
- Confusion matrices for misclassification analysis

### Deep learning: EEGNet

**Architecture:** EEGNet (Lawhern et al. 2018) — a compact CNN designed specifically for EEG.
- Temporal conv → depthwise spatial conv → separable conv → classification head
- ~2k parameters; trains fast even on CPU

**Training:**
- Per-subject training with train/val split
- Cross-entropy loss, Adam optimizer
- Early stopping on validation accuracy

**Why compare both:**
- Classical (CSP+LDA) is the established BCI baseline
- EEGNet represents the modern deep-learning approach
- The comparison itself is a key portfolio talking point

---

## 6. Dashboard

**Stack:** Streamlit
**Purpose:** Let any reviewer pick a subject and see preprocessing → ERP → time-frequency → classification results without running notebooks.

**Pages:**
1. Subject selector + raw signal preview
2. Preprocessing inspection (before/after, ICA components)
3. ERP & topomaps
4. Time-frequency plots
5. Classification results & confusion matrices

---

## References

- Schalk et al. (2004). BCI2000: A general-purpose brain-computer interface system. *IEEE Trans Biomed Eng.*
- Lawhern et al. (2018). EEGNet: A compact convolutional neural network for EEG-based BCIs. *J Neural Eng.*
- Pfurtscheller & Lopes da Silva (1999). Event-related EEG/MEG synchronization and desynchronization. *Clin Neurophysiol.*
- Blankertz et al. (2008). Optimizing spatial filters for robust EEG single-trial analysis. *IEEE Sig Proc Mag.*
