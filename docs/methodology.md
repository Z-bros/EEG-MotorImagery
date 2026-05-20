# Methodology

Detailed methodology writeup for the EEG Motor Imagery Classification project.
Complements the chunk-by-chunk notebooks (`notebooks/01..06`) and the
top-level [README](../README.md). Aimed at a technical reader who wants to
understand the *why* behind each design choice, not just the *what*.

---

## Table of Contents

1. [Dataset and task](#1-dataset-and-task)
2. [Preprocessing pipeline](#2-preprocessing-pipeline)
3. [Validation strategy](#3-validation-strategy)
4. [Methods compared](#4-methods-compared)
5. [Spatial filter interpretability](#5-spatial-filter-interpretability)
6. [Key methodological decisions and their justifications](#6-key-methodological-decisions-and-their-justifications)
7. [What we deliberately did *not* do, and why](#7-what-we-deliberately-did-not-do-and-why)
8. [References](#8-references)

---

## 1. Dataset and task

**Source:** PhysioNet EEGBCI dataset (Schalk et al. 2004; Goldberger et al.
2000). 109 subjects, each performing motor imagery and motor execution tasks
in separate runs.

**Task selected:** Binary classification of **T1 (imagined left fist)** vs
**T2 (imagined right fist)**, drawn from runs 4, 8, and 12 (the three
imagery runs for hand movement). Approximately 22–28 trials per class per
subject, before quality control.

**Sampling rate:** 160 Hz for 106 subjects; 128 Hz for 3 outlier subjects.
The 3 outliers are upsampled to 160 Hz during multi-subject concatenation
(rather than downsampling the 106-strong majority).

**Why this dataset:** Open access, large enough for population-scale
analysis (109 subjects), widely benchmarked in the BCI literature
(so our results have a comparison frame), and complete with a documented
experimental paradigm (cue-based motor imagery, eyes-open, seated).

**Why binary T1 vs T2 specifically:** The cleanest motor imagery contrast
on this dataset. The four-class formulation (T1/T2 across imagery and
execution) introduces additional confounds. Binary is sufficient to
demonstrate the methodological points and reproduces published baselines
cleanly.

---

## 2. Preprocessing pipeline

Implemented in `src/preprocessing.py`. Composable functions chained in
`preprocess_subject(subject, runs=(4,8,12))`. Pipeline:

1. **Load** raw EDF via `mne.io.read_raw_edf`.
2. **Standardize channel names** to remove EDF-specific suffixes ("..", etc.)
   and apply the **standard_1005 montage** for spatial position information.
3. **Notch filter 60 Hz** to remove power-line interference (PhysioNet
   recordings are US-sourced).
4. **Bandpass 1–40 Hz** to retain delta through low gamma. Cuts very
   low-frequency drift (below 1 Hz) and high-frequency muscle artifact /
   line harmonics (above 40 Hz).
5. **Statistical bad-channel detection.** Flag channels with z-scored
   variance or kurtosis exceeding threshold. Interpolate rather than drop
   — preserves feature dimensionality across subjects, which matters for
   downstream multi-subject concatenation and fixed-input deep models.
6. **Average reference.** Re-reference to the average across channels.
7. **ICA with Picard algorithm**, 20 components, using Fp1 as a pseudo-EOG
   channel for ocular component detection. Excluded components are
   typically blinks (textbook ICA000), saccades, and occasionally heartbeat.
8. **Epoch** from −0.5 to +4.0 s relative to cue onset, baseline corrected
   on the [-0.5, 0] window.
9. **AutoReject** for automated trial-level rejection and channel-by-trial
   interpolation. Used in `cv` mode for unbiased threshold selection.

**Why each choice:**

- *1–40 Hz bandpass, not 8–30 Hz*: We don't want to commit the entire pipeline
  to mu/beta range. The cluster permutation tests in Chunk 3 wanted to see
  the full ERP, and the classical features in Chunk 4 wanted full-spectrum
  band power. Sub-band filtering is applied per-feature, not at the
  pipeline level.
- *20 ICA components, not 64*: Computational tractability and stability.
  With ~45 trials per subject after AutoReject, full-rank ICA overfits.
  20 captures the major artifact sources (blinks, saccades, cardiac,
  EMG bursts) without slicing into the neural signal.
- *Interpolate bads, not drop*: Cross-subject analyses require constant
  channel dimensionality. Dropping bad channels per-subject breaks this.
  Interpolation introduces some smoothing artifact but preserves the
  comparability we need.
- *Modal sfreq for concatenation*: Upsampling 3 subjects from 128 to 160 Hz
  is cheaper than downsampling 106 from 160 to 128 Hz, and preserves more
  information overall.

**Quality control on the pipeline:**

- Subject 1 retains ~92% of epochs (Chunk 4 measurement) — well within the
  typical 70–95% retention range for motor imagery data after AutoReject.
- ICA000 is consistently a textbook blink component (frontal topography,
  blink-shaped time course) across subjects, confirming the ocular component
  is being identified correctly.

---

## 3. Validation strategy

Two CV regimes throughout, with deliberate methodological choices:

### Within-subject CV

5-fold stratified K-fold per subject (so each fold has roughly balanced
T1/T2). Each subject is its own analysis. Aggregated by taking the **median
of per-subject mean balanced_accuracy** across the 109 subjects.

For Chunk 5 classical work: 5 folds × 10 repeats per subject.
For Chunk 6 deep learning: 5 folds × 1 repeat per subject (GPU time budget;
10 repeats would have been 15+ hours).

### Cross-subject CV

5-fold **GroupKFold** with subject_id as the group. Each test fold contains
~22 disjoint subjects. The model never sees any trial from a test subject
during training.

This is the methodologically correct way to estimate subject-independent
generalization — what a deployed BCI would face when used by a new user
who hasn't provided calibration data.

### Primary metric: balanced_accuracy

Four subjects had ~1.5× class imbalance after AutoReject (i.e., AutoReject
rejected proportionally more T1 or T2 trials). Plain accuracy would have
been inflated for the majority class on those subjects. Balanced accuracy
is the macro-average of per-class recall and is robust to this.

### What we do *not* do

- **Trial-randomized splits** ignore subject identity and cause severe
  leakage on this dataset (trials from the same subject in train and test).
  Papers using trial-randomized splits report 95–99% accuracy on this data;
  the same models under GroupKFold drop to 0.70–0.85. The leakage is well-
  documented (Lotte et al. 2018 review). We avoid it.
- **Peak-not-mean reporting.** Some papers report the best-performing CV
  fold or the best run across multiple random seeds. We report median across
  subjects.
- **Cherry-picked responder subsets.** Some papers report performance on a
  hand-picked subset of subjects with strong motor imagery responses.
  Our 109-subject median includes the ~50% non-responder fraction.

---

## 4. Methods compared

### Classical (Chunks 4 + 5)

Implemented in `src/features.py` and `src/multisubject.py`.

- **Lateralization indices (Lat) + LogReg.** Compute the asymmetry of mu
  and beta band power between left and right motor channels (C3 vs C4,
  CP3 vs CP4, FC3 vs FC4). 18-dimensional feature per epoch. Best
  classical baseline at 0.560 cross-subject.

- **Motor band power (BP) + LinearSVC.** Average mu (8–13 Hz) and beta
  (13–30 Hz) power over motor channels. 6-dimensional feature per epoch.
  Slightly behind Lat at 0.555 cross-subject.

- **Common Spatial Patterns (CSP) + LDA.** Standard CSP with 4 components.
  Failed: 0.501 cross-subject, no better than chance. Diagnostic in
  Chunk 5 §6 revealed CSP filters localized to frontal/anterior-temporal
  channels (Fp1, Fp2, F7, F8, T8), not motor cortex — residual EMG/EOG
  contamination that survived 20-component ICA.

### Deep learning (Chunk 6)

Implemented in `src/eegnetmods.py`. PyTorch.

- **EEGNet (Lawhern et al. 2018).** ~2.8K params at our input shape.
  Depthwise-separable conv architecture: temporal filter → depthwise spatial
  filter → separable conv → linear. Max-norm constraints on depthwise
  spatial and classifier layers. Cross-subject 0.758, within-subject 0.570.

- **ShallowConvNet (Schirrmeister et al. 2017).** ~107K params. "Deep CSP"
  design: temporal conv → spatial conv → square → average pool → log →
  linear. Cross-subject **0.802** (the project's best result), within-subject
  0.565.

### Training protocol (deep learning)

- Adam, lr=1e-3, batch size 64
- Max 100 epochs, early stopping (patience 15, on inner-val balanced_accuracy)
- 85/15 stratified inner train/val split for early stopping signal
- Per-channel z-score normalization fit on training fold only
- Class weights from training fold class counts
- Training-only augmentation: channel dropout (p=0.1), random time crop to
  90% with zero-pad, Gaussian noise (σ = 0.1× un-normalized channel std)
- Deterministic seeds (numpy, torch, CUDA) per fold

The augmentation σ is scaled by *un-normalized* channel std before
normalization, so the perturbation is on the order of 10% of natural channel
variance (its intended physical meaning) rather than 10% of unit variance
(which would be much smaller post-normalization).

---

## 5. Spatial filter interpretability

The most distinctive methodological contribution of this project is the
**topomap analysis of learned filters**, paralleling the classical CSP
filter analysis from Chunk 5 §6.

### The question

CSP failed at 0.501 cross-subject. Chunk 5 §6 diagnosed why: population-
averaged CSP filters localized to frontal-pole and anterior-temporal channels
rather than sensorimotor cortex. The classical algorithm was learning
artifact, not signal.

For Chunk 6: did EEGNet and ShallowConvNet avoid this failure mode? Or did
they succeed at the task while still relying on the same contaminated
features?

### How

Each architecture exposes a "spatial conv" layer whose weights are
interpretable as scalp topomaps (one weight per channel per filter). For
each cross-subject fold's trained model, we:

1. Extract the spatial conv weights (`get_spatial_filters()` method on the
   model).
2. Run the training data through the frozen model, capturing per-filter
   activation magnitude. Cached as `filter_activations_` on the fitted
   classifier.
3. Sign-align across the 5 cross-subject folds (spatial filters have
   arbitrary sign — the network can flip both the filter and the next layer
   without changing output, so naive averaging would cancel).
4. Average across folds and rank by activation magnitude.
5. Plot top-6 as topomaps. Compute % of total |weight| concentrated in
   motor ROI, frontal ROI, and temporal ROI.

ShallowConvNet's spatial conv weights are `(40 spatial filters × 40 temporal
filters × 64 channels)`. We aggregate over the temporal-filter axis using
RMS to produce one `(40, 64)` map per fold. This collapses dipolar signed
structure into magnitude only — visually less crisp but quantitatively
correct for "where does this filter attend on the scalp."

### Findings (Chunk 6 §8)

**EEGNet:** 28% average motor-ROI weight, 17% frontal. Top-3 channels per
filter dominated by F8, AF8, F7, FT7, AF7. **Replicates classical CSP's
failure mode.**

**ShallowConvNet:** 22% motor, 18% frontal. Top-3 channels dominated by
C3, C1, CP3, CP4, FCz, C4. **Avoided the contamination.**

### Why ShallowConvNet avoids it

The `square → log → average pool` block transforms raw signal into log
band-power over a sliding window. Broadband EMG and DC drift are suppressed
by this transformation relative to tight-band mu/beta oscillations. The
architecture's domain assumption ("the discriminative signal lives in log
band power") steers gradient descent toward filters that pick up neuronal
oscillations rather than artifact channels.

EEGNet has no analogous structural constraint — its depthwise spatial conv
finds whatever channel mixture predicts label, and on this dataset that
includes residual EMG/EOG correlates with T1 vs T2 (anticipatory eye
movement, motor-imagery-correlated facial tension).

So Shallow's 4-point accuracy advantage is a *quality* difference, not just
a magnitude difference. Shallow is using the physiologically meaningful
features the protocol was designed to elicit; EEGNet is partly exploiting
correlated artifact.

---

## 6. Key methodological decisions and their justifications

A consolidated list of decisions made across the chunks, each with a one-line
justification:

| Decision | Justification |
|---|---|
| Balanced accuracy as primary metric | 4 subjects had ~1.5× class imbalance after AutoReject |
| Interpolate bad channels (not drop) | Cross-subject analyses require constant channel dimensionality |
| GroupKFold for cross-subject CV | Subject identity is the leakage source — must be the grouping key |
| Modal sfreq (160 Hz) for concatenation | Cheaper than downsampling the 106-subject majority |
| 5-fold × 1 repeat for deep CV (not × 10) | GPU time budget; 10 repeats would be 15+ hours |
| Hardcoded vs CSV-loaded Chunk 5 baselines | These are frozen comparison numbers; hardcode for self-contained reproducibility |
| RMS aggregation for Shallow filters | Spatial filters have arbitrary sign; need magnitude-only for stable averaging |
| Augmentation σ scaled by pre-normalization std | Preserves "10% of natural channel variance" semantic across normalization |
| `torch.use_deterministic_algorithms(True, warn_only=True)` | T4 may not have deterministic kernels for every op; warn rather than fail |
| Skip Chunk 5 §6 CSP filter regeneration for §8 overlay | Findings already documented; regeneration adds time without new information |

---

## 7. What we deliberately did *not* do, and why

Negative methodological choices are as important as positive ones for a
portfolio reader assessing rigor.

### Not done: nested CV with hyperparameter search

Attempted in Chunk 5 §5 but skipped due to OOM on 32 GB RAM. Documented in
that notebook's §5 markdown rather than refactored. Justification:
hyperparameter tuning at population scale would only help if our methods
were near a tuning ceiling, but at N=3900 training trials per cross-subject
fold the 6- and 18-dimensional classical features have negligible C
sensitivity, and CSP n_components tuning doesn't fix filter contamination.

### Not done: transfer learning

Originally scoped out of Chunk 6 for time reasons. Pre-training on cross-
subject data and fine-tuning per subject is a natural way to break the
within-subject ceiling, but doubles the experimental complexity. Parked
as a follow-up.

### Not done: Riemannian / filter-bank features

Filter-bank CSP and Riemannian features (using the geometry of SPD matrices)
are state-of-the-art for classical motor-imagery BCI. We omit them
deliberately — the project's classical baselines are intentionally simple
to make the deep-learning comparison cleaner. Riemannian methods would
likely close some of the classical-vs-DL gap on cross-subject without
changing the qualitative story.

### Not done: peak performance reporting

The deep learning literature on PhysioNet motor imagery has many papers
reporting peak accuracy across CV folds or peak across multiple seeds.
We report median throughout. This is more conservative but more honest.

### Not done: cherry-picked responder subsets

Our 109-subject median includes the ~50% non-responder fraction (subjects
with no detectable lateralization signature at the individual level).
Reporting only responders would inflate cross-subject median by 0.05-0.10
but mischaracterize the practical reproducibility.

### Not done: subject-specific calibration

Real BCIs typically include a 10–30 trial calibration phase per user. Our
cross-subject model is *fully* user-independent — no calibration assumed.
This makes the numbers lower than calibrated BCIs but more directly relevant
to "how well does a pretrained model generalize cold?"

---

## 8. References

### Foundational EEG / BCI

- Schalk, G., et al. (2004). BCI2000: A General-Purpose Brain-Computer
  Interface (BCI) System. *IEEE TBME* 51(6):1034-1043.
- Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
  *Circulation* 101(23):e215-e220.
- Pfurtscheller, G., & Lopes da Silva, F. H. (1999). Event-related EEG/MEG
  synchronization and desynchronization: basic principles. *Clinical
  Neurophysiology* 110(11):1842-1857.
- Wolpaw, J. R., et al. (2002). Brain-computer interfaces for communication
  and control. *Clinical Neurophysiology* 113(6):767-791.

### Methods

- Lotte, F., et al. (2018). A review of classification algorithms for
  EEG-based brain-computer interfaces: a 10 year update. *J. Neural
  Engineering* 15(3):031005.
- Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural
  network for EEG-based brain-computer interfaces. *J. Neural Engineering*
  15(5):056013.
- Schirrmeister, R. T., et al. (2017). Deep learning with convolutional
  neural networks for EEG decoding and visualization. *Human Brain Mapping*
  38(11):5391-5420.

### Tooling

- Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python.
  *Frontiers in Neuroscience* 7:267.
- Jas, M., et al. (2017). Autoreject: Automated artifact rejection for MEG
  and EEG data. *NeuroImage* 159:417-429.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.
  *JMLR* 12:2825-2830.
- Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance
  Deep Learning Library. *NeurIPS* 32:8024-8035.
