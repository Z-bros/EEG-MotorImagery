"""
Feature extraction for EEG motor imagery analysis.

This module computes numerical representations of EEG epochs:
- Event-related potentials (ERPs): trial-averaged time-domain signals
- Time-frequency representations (TFRs): power/phase as a function of
  time and frequency, where motor imagery's true signal lives (mu/beta ERD)

Functions return MNE objects (Evoked, AverageTFR) or ndarrays — never figures.
Plotting lives in src/visualization.py; statistical testing lives in src/stats.py.

This separation means notebook 04 (feature extraction for ML) can reuse
compute_tfr_morlet() without dragging matplotlib into a feature pipeline.
"""

from typing import Optional, Union, List

import numpy as np
import mne
from mne.time_frequency import tfr_morlet

def compute_evoked(
    epochs: mne.Epochs,
    condition: str,
    picks: Optional[Union[str, List[str]]] = None,
    return_epochs: bool = False,
) -> mne.Evoked:
    """
    Compute the trial-averaged ERP for a single condition.

    Parameters
    ----------
    epochs : mne.Epochs
        Cleaned epochs from preprocessing (notebook 02). Must contain the
        requested condition in its event_id mapping.
    condition : str
        Event label, e.g. 'T1' (left fist imagery) or 'T2' (right fist
        imagery). Must match a key in epochs.event_id.
    picks : str | list | None
        Channel selection forwarded to epochs.average(). None averages
        across all good EEG channels.

    Returns
    -------
    evoked : mne.Evoked
        The mean across trials. nave attribute reports trial count, which
        the caller should check — with ~45 trials/condition for one subject,
        SNR will be modest and we expect a noisy ERP. That's the point of
        this notebook: showing honestly what averaging buys us.

    Notes
    -----
    We use the arithmetic mean (MNE default) rather than a robust estimator
    like the trimmed mean. With AutoReject already having removed extreme
    epochs upstream, the residual distribution should be approximately
    Gaussian, so the mean is the maximum-likelihood estimator. If you later
    skip AutoReject (e.g., for speed when scaling to 109 subjects), revisit
    this — `method='median'` is available on epochs.average().
    """
    if condition not in epochs.event_id:
        raise ValueError(
            f"Condition '{condition}' not in epochs.event_id "
            f"(available: {list(epochs.event_id.keys())})"
        )

    epochs_subset = epochs[condition].copy()
    evoked = epochs_subset.average(picks=picks)
    evoked.comment = condition

    if return_epochs:
        return evoked, epochs_subset
    return evoked


def compute_evoked_contrast(
    epochs: mne.Epochs,
    condition_a: str = 'T1',
    condition_b: str = 'T2',
) -> mne.Evoked:
    """
    Compute the difference wave (condition_a − condition_b).

    Difference waves cancel components common to both conditions (cue
    perception, generic attention, baseline drifts) and isolate what
    distinguishes them. For motor imagery, the time-domain difference is
    expected to be small — the lateralization story is much clearer in
    the frequency domain — but worth showing for completeness.

    Parameters
    ----------
    epochs : mne.Epochs
        Cleaned epochs.
    condition_a, condition_b : str
        Event labels. Difference is computed as a − b.

    Returns
    -------
    evoked_diff : mne.Evoked
        Difference wave. The `comment` attribute is set to
        f"{condition_a} - {condition_b}" for downstream plotting clarity.
        The `nave` attribute is set to the harmonic mean of the two
        contributing nave values (MNE's combine_evoked convention for
        difference waves — this is the effective N for noise-level
        calculations on the difference).
    """
    evoked_a = compute_evoked(epochs, condition=condition_a)
    evoked_b = compute_evoked(epochs, condition=condition_b)

    # combine_evoked with weights [1, -1] computes the difference and
    # handles nave bookkeeping properly. Manually subtracting .data would
    # leave nave wrong, which would mess up any downstream variance-based
    # plotting (confidence bands, GFP, etc.).
    evoked_diff = mne.combine_evoked([evoked_a, evoked_b], weights=[1, -1])
    evoked_diff.comment = f"{condition_a} - {condition_b}"

    return evoked_diff


def compute_tfr_morlet(
    epochs: mne.Epochs,
    freqs: Optional[np.ndarray] = None,
    n_cycles: Optional[Union[float, np.ndarray]] = None,
    condition: Optional[str] = None,
    decim: int = 3,
    average: bool = True,
    return_itc: bool = False,
    n_jobs: int = 1,
):
    """
    Compute time-frequency representation via Morlet wavelets.

    This is the headline computation of the notebook. Motor imagery's
    signature is event-related desynchronization (ERD) in the mu band
    (8–13 Hz) and beta band (13–30 Hz) over sensorimotor cortex,
    contralateral to the imagined movement. Morlet wavelets are the
    standard tool: each frequency is convolved with a Gaussian-windowed
    sinusoid, giving an explicit time-frequency-resolution tradeoff we
    control via n_cycles.

    Parameters
    ----------
    epochs : mne.Epochs
        Cleaned epochs.
    freqs : ndarray | None
        Frequencies to estimate (Hz). Defaults to np.arange(4, 36, 1) —
        4 Hz lower bound stays clear of our 1 Hz highpass edge (we don't
        trust power estimates within a few Hz of the filter cutoff);
        35 Hz upper bound stays below the 40 Hz lowpass edge for the same
        reason; 1 Hz resolution is fine-grained enough to resolve mu
        (8–13 Hz) and beta (13–30 Hz) sub-bands distinctly.
    n_cycles : float | ndarray | None
        Number of wavelet cycles per frequency. Defaults to freqs / 2.
        This is a deliberate frequency-dependent choice rather than a
        constant: at low frequencies (4 Hz), 2 cycles is 500 ms — enough
        to localize ERD onset in time. At high frequencies (35 Hz), ~17
        cycles is ~485 ms — giving good frequency resolution where
        spectral structure is finer. A constant n_cycles=7 (also common)
        would over-localize in time at low freqs and over-smooth at high
        freqs. The freqs/2 rule is a good middle ground; tighten to
        freqs/3 if you need sharper time resolution and can accept
        smearing across mu/beta.
    condition : str | None
        If provided, subset epochs to this condition before computing.
        If None, compute on all epochs (rarely what you want — usually
        you want per-condition TFRs to contrast).
    decim : int
        Temporal downsampling factor applied after wavelet convolution.
        decim=3 takes 160 Hz → ~53 Hz, which is plenty for visualizing
        ERD dynamics (mu-band envelope changes on ~100 ms timescales,
        Nyquist of 26 Hz easily captures this) and cuts memory ~3×.
        Set to 1 if you later want to feed TFR samples directly to a
        classifier and need full temporal resolution.
    average : bool
        If True (default), average across trials → AverageTFR. If False,
        return per-trial EpochsTFR — needed for single-trial features
        (chunk 4) and for cluster permutation tests (which need trials).
    return_itc : bool
        Inter-trial coherence. Not the primary signal for motor imagery
        (ERD is amplitude, not phase) so default False, but cheap to
        compute alongside if you want it. NOTE: only valid when
        average=True; tfr_morlet enforces this.
    n_jobs : int
        Parallel jobs. Start with 1 for one subject; bump for 109.

    Returns
    -------
    tfr : AverageTFR or EpochsTFR
        Power values are raw (V²/Hz-like units). Apply baseline correction
        via apply_tfr_baseline() before plotting.
        If return_itc=True and average=True, returns (power, itc) tuple.
    """
    if freqs is None:
        # 4 Hz floor: stays clear of the 1 Hz highpass filter's transition
        # band (rule of thumb: don't trust power within ~3× the highpass
        # edge). 35 Hz ceiling: same reasoning for the 40 Hz lowpass.
        freqs = np.arange(4, 36, 1)
    else:
        freqs = np.asarray(freqs)

    if n_cycles is None:
        # freqs/2 gives roughly constant wavelet duration (~500 ms) across
        # the band, which is appropriate for ERD which evolves on
        # similar timescales at mu and beta frequencies.
        n_cycles = freqs / 2.0

    # Subset to condition if requested. Use .copy() so we don't mutate
    # the caller's epochs object.
    if condition is not None:
        if condition not in epochs.event_id:
            raise ValueError(
                f"Condition '{condition}' not in epochs.event_id "
                f"(available: {list(epochs.event_id.keys())})"
            )
        epochs_to_use = epochs[condition].copy()
    else:
        epochs_to_use = epochs

    # use_fft=True: standard for Morlet — convolution via FFT is faster
    # for the typical # of cycles we're using.
    # return_itc: forced False if average=False (tfr_morlet would raise).
    tfr = tfr_morlet(
        epochs_to_use,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=return_itc if average else False,
        decim=decim,
        average=average,
        n_jobs=n_jobs,
    )

    return tfr


def apply_tfr_baseline(
    tfr,
    baseline: tuple = (-0.5, -0.1),
    mode: str = 'logratio',
):
    """
    Baseline-correct a TFR in place and return it.

    Parameters
    ----------
    tfr : AverageTFR | EpochsTFR
        Computed via compute_tfr_morlet.
    baseline : tuple of float
        (tmin, tmax) in seconds, relative to event onset. Default
        (-0.5, -0.1): uses the full pre-cue window but stops 100 ms
        before t=0 to avoid contamination from any anticipatory activity
        right at cue onset. Our epoch starts at -0.5 so this uses the
        entire available pre-stimulus baseline.
    mode : str
        'logratio' (default): 10*log10(power / baseline_mean). Symmetric
        around 0, interpretable as dB change, handles the fact that EEG
        power is log-normally distributed. ERD shows as negative dB,
        ERS as positive — the convention in the motor imagery literature.
        Alternatives: 'percent' (linear, less common in this domain),
        'zscore' (good for stats but harder to interpret physiologically),
        'mean' (subtract baseline mean — only for already-log data).

    Returns
    -------
    tfr : same type as input
        Modified in place AND returned, for chaining.
    """
    tfr.apply_baseline(baseline=baseline, mode=mode)
    return tfr


def get_motor_channels() -> dict:
    """
    Return the canonical sensorimotor channel groupings for this dataset.

    The PhysioNet motor imagery dataset uses 10-10 montage. The
    sensorimotor channels of interest are:
    - C3: left hemisphere, contralateral to right-hand movement
    - C4: right hemisphere, contralateral to left-hand movement
    - Cz: midline, supplementary motor area
    Plus their immediate neighbors for cluster-level analysis.

    Returns
    -------
    channels : dict
        Keys: 'left_motor', 'right_motor', 'midline', 'all_motor'.
        Values: lists of channel names. Use 'left_motor' when analyzing
        right-fist imagery (T2) and 'right_motor' for left-fist (T1) —
        this is the contralateral expectation.

    Notes
    -----
    Centralizing this here means notebooks don't hardcode ['C3', 'C4']
    in multiple places, and if you later switch montages or want to
    expand the ROI, you change it once.

    Channel names follow MNE's normalized convention from
    preprocessing.standardize_channel_names() — capital first letter,
    lowercase second character, no periods.
    """
    return {
        # Left hemisphere motor cluster — activated for right-hand imagery
        'left_motor': ['C3', 'C1', 'C5', 'FC3', 'CP3'],
        # Right hemisphere motor cluster — activated for left-hand imagery
        'right_motor': ['C4', 'C2', 'C6', 'FC4', 'CP4'],
        # Midline — supplementary motor area, bilateral movements
        'midline': ['Cz', 'FCz', 'CPz'],
        # Combined ROI for omnibus analyses
        'all_motor': [
            'C3', 'C1', 'C5', 'FC3', 'CP3',
            'C4', 'C2', 'C6', 'FC4', 'CP4',
            'Cz', 'FCz', 'CPz',
        ],
    }


# =============================================================================
# === ML features  ============================================================
# =============================================================================
#
# The functions above (compute_evoked, compute_tfr_morlet, etc.) produce
# quantities for *analysis and visualization* — MNE objects designed for
# plotting and statistical testing. The functions below produce quantities
# for *classifier input* — (n_epochs, n_features) numpy arrays plus aligned
# label vectors.
#
# Three feature families are implemented, in increasing order of complexity:
#
# 1. Motor-channel band power: log-power in mu (8-13 Hz) and beta (13-30 Hz)
#    bands, restricted to ~9 sensorimotor channels. The interpretable
#    baseline — when this works, you can point at C3 and say what's happening.
#    18 features total (9 channels × 2 bands).
#
# 2. Lateralization indices: normalized C3-vs-C4 power differences per band.
#    Directly encodes the quantity that Chunk 3 showed is class-discriminative
#    in this subject. 4 features total (2 bands × 2 channel pairs, or fewer
#    depending on configuration).
#
# 3. CSP log-variance: learned spatial filters maximizing class variance ratio.
#    Standard MI-BCI feature. 4-8 features.
#
# All three return arrays in (n_epochs, n_features) layout aligned with a
# label vector y from epochs.events[:, -1]. Stateless extractors (band power,
# lateralization) are functions; the stateful one (CSP) is a sklearn-compatible
# transformer so it can be dropped into Pipelines without leakage.
#
# Standardization and scaling live in the classifier Pipeline in notebook 04,
# NOT here. These functions return raw log-power and log-variance.

from dataclasses import dataclass, field
from typing import Literal, Optional, Union
import numpy as np
import mne
from mne.decoding import CSP as MNE_CSP
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Frequency bands for MI classification.
# Note: Chunk 3 found the strongest ERD in 10-15 Hz specifically, narrower
# than canonical mu (8-13). We use canonical bands here as the default for
# methodological transparency (matches the literature) but expose `bands`
# as a parameter so notebook 04 can compare canonical vs. data-driven bands
# as a hyperparameter sweep.
MI_BANDS_ML: dict[str, tuple[float, float]] = {
    "mu":   (8.0, 13.0),
    "beta": (13.0, 30.0),
}

# Sensorimotor channels (10-05 montage names, post-Chunk-2 standardization).
# These nine cover the central sulcus hand-area neighborhood: surrounding
# C3/C4 with FC and CP rings because individual hand representations vary
# by a few cm around the canonical site.
SENSORIMOTOR_CHANNELS_ML: list[str] = [
    "FC3", "FCz", "FC4",
    "C3",  "Cz",  "C4",
    "CP3", "CPz", "CP4",
]

# Lateralized channel pairs for lateralization-index features.
# Each tuple is (left_hemisphere_channel, right_hemisphere_channel).
# C3/C4 is the canonical pair; FC3/FC4 and CP3/CP4 add anterior/posterior
# coverage in case the subject's hand representation is shifted from
# precentral gyrus proper.
LATERALIZATION_PAIRS: list[tuple[str, str]] = [
    ("C3",  "C4"),
    ("FC3", "FC4"),
    ("CP3", "CP4"),
]


@dataclass
class MLFeatureSet:
    """Container bundling ML features, labels, and metadata.

    Named MLFeatureSet (rather than FeatureSet) to avoid colliding with
    any analysis-side container the Chunk 3 visualization code might use.
    Carrying metadata alongside (X, y) lets downstream permutation
    importance, feature-correlation heatmaps, and CV reports introspect
    column meaning without rederiving order from convention.

    Attributes
    ----------
    X : np.ndarray, shape (n_epochs, n_features)
    y : np.ndarray, shape (n_epochs,)
    feature_names : list[str]
        Human-readable name per column. E.g. "C3_mu_logpow", "LI_mu_C3-C4",
        "CSP1".
    metadata : dict
        Extraction parameters (bands, time window, channel selection, etc.)
        for reproducibility and figure captions.
    """
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {self.X.shape}")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"X and y first dim must match: "
                f"{self.X.shape[0]} vs {self.y.shape[0]}")
        if self.X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"feature_names length ({len(self.feature_names)}) "
                f"must equal n_features ({self.X.shape[1]})")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _select_channels(
    epochs: mne.Epochs,
    ch_names: list[str],
    strict: bool = False,
) -> mne.Epochs:
    """Subset epochs to a named channel list.

    If `strict=False` (default), silently drops requested channels that
    aren't in the data — useful because EEGBCI files occasionally have
    one or two channels excluded after bad-channel rejection in Chunk 2.
    If `strict=True`, raises on any missing channel.

    Returns a copy; does not mutate the caller's epochs.
    """
    available = [c for c in ch_names if c in epochs.ch_names]
    missing = [c for c in ch_names if c not in epochs.ch_names]
    if missing:
        if strict:
            raise ValueError(f"Channels not in epochs: {missing}")
        # else: caller will get whatever's available. The metadata dict
        # records what was actually used so this isn't silently lost.
    if not available:
        raise ValueError(
            f"None of the requested channels {ch_names} are in epochs. "
            f"Available channels: {epochs.ch_names[:10]}...")
    return epochs.copy().pick(available)


def _welch_band_power(
    epochs: mne.Epochs,
    fmin: float,
    fmax: float,
    tmin: Optional[float],
    tmax: Optional[float],
    log: bool = True,
) -> np.ndarray:
    """Per-channel mean PSD within a frequency band.

    Returns
    -------
    power : np.ndarray, shape (n_epochs, n_channels)
        log10(power) if log=True, else raw power.

    Implementation notes
    --------------------
    - Welch's method via `epochs.compute_psd(method='welch', ...)`. For
      a 3 s window at 160 Hz this gives ~0.33 Hz frequency resolution,
      sufficient to resolve mu from neighboring alpha/beta.
    - Power averaged across frequencies WITHIN the band (not integrated).
      Mean vs. integral differs by a constant bandwidth factor that
      cancels under log-transform and standardization, so the choice
      doesn't affect classification.
    - Log10 guards against zeros with a 1e-30 floor. Real EEG won't hit
      this but a defensive max() costs nothing.
    """
    spectrum = epochs.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        tmin=tmin,
        tmax=tmax,
        verbose=False,
    )
    psd = spectrum.get_data()         # (n_epochs, n_channels, n_freqs)
    power = psd.mean(axis=-1)         # (n_epochs, n_channels)
    if log:
        power = np.log10(np.maximum(power, 1e-30))
    return power


# ---------------------------------------------------------------------------
# Feature family 1: motor-channel band power
# ---------------------------------------------------------------------------

def compute_motor_bandpower(
    epochs: mne.Epochs,
    bands: dict[str, tuple[float, float]] = MI_BANDS_ML,
    channels: list[str] = SENSORIMOTOR_CHANNELS_ML,
    tmin: float = 0.5,
    tmax: float = 3.5,
    log: bool = True,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Log band-power features on sensorimotor channels.

    The interpretable baseline. Restricts to ~9 motor channels (rather
    than all 64) for two reasons:

    1. **Feature-to-sample ratio.** With ~44 epochs total, 64 channels ×
       2 bands = 128 features creates a p >> n problem that even strong
       regularization can't fully fix. 9 × 2 = 18 features is tractable.

    2. **Prior knowledge from Chunk 3.** TFR analysis localized the mu
       ERD signal to motor cortex. Including occipital and frontal
       channels mostly adds noise; we already know the signal isn't
       there.

    Why (0.5, 3.5) s window: skip the 500 ms ERD onset transient,
    skip the last 500 ms to avoid end-of-trial contamination. Leaves
    3 s of stationary imagery — 24 cycles of 8 Hz mu, plenty for stable
    Welch estimates.

    Why log: band power is right-skewed (variance ratios span orders
    of magnitude); log-power is roughly Gaussian, which is what linear
    classifiers (LDA, LogReg) assume. Standard MI convention since
    Pfurtscheller & Neuper (2001).

    Parameters
    ----------
    epochs : mne.Epochs
        Cleaned epochs from Chunk 2.
    bands : dict[str, (fmin, fmax)]
        Frequency bands. Default: canonical mu and beta.
    channels : list[str]
        Sensorimotor channel names. Channels not in epochs are silently
        dropped; the returned `channels_used` reports what was actually
        included.
    tmin, tmax : float
        Time window for PSD estimation.
    log : bool
        If True, return log10(power). Strongly recommended for linear
        classifiers.

    Returns
    -------
    X : np.ndarray, shape (n_epochs, n_channels_used * n_bands)
        Channel-major column order: [ch0_band0, ch0_band1, ch1_band0, ...].
    feature_names : list[str]
        E.g. ["C3_mu_logpow", "C3_beta_logpow", "Cz_mu_logpow", ...].
    channels_used : list[str]
        Channels actually included (subset of requested if any were missing).
    """
    if not bands:
        raise ValueError("`bands` must contain at least one band")

    sub = _select_channels(epochs, channels, strict=False)
    channels_used = sub.ch_names

    n_epochs = len(sub)
    n_channels = len(channels_used)
    n_bands = len(bands)

    # Compute each band's per-channel power once.
    band_to_power: dict[str, np.ndarray] = {}
    for band_name, (fmin, fmax) in bands.items():
        band_to_power[band_name] = _welch_band_power(
            sub, fmin, fmax, tmin, tmax, log=log)

    # Channel-major interleave.
    X = np.empty((n_epochs, n_channels * n_bands), dtype=np.float64)
    feature_names: list[str] = []
    band_list = list(bands)
    suffix = "logpow" if log else "pow"
    for ch_idx, ch_name in enumerate(channels_used):
        for band_idx, band_name in enumerate(band_list):
            col = ch_idx * n_bands + band_idx
            X[:, col] = band_to_power[band_name][:, ch_idx]
            feature_names.append(f"{ch_name}_{band_name}_{suffix}")

    return X, feature_names, channels_used


# ---------------------------------------------------------------------------
# Feature family 2: lateralization indices
# ---------------------------------------------------------------------------

def compute_lateralization_indices(
    epochs: mne.Epochs,
    bands: dict[str, tuple[float, float]] = MI_BANDS_ML,
    pairs: list[tuple[str, str]] = LATERALIZATION_PAIRS,
    tmin: float = 0.5,
    tmax: float = 3.5,
    use_log_power: bool = False,
) -> tuple[np.ndarray, list[str], list[tuple[str, str]]]:
    """Lateralization indices: normalized inter-hemisphere power differences.

    For each (left_ch, right_ch) pair and each band, computes:

        LI = (P_right - P_left) / (P_right + P_left)

    Range: [-1, +1]. Positive = more power on the right; negative = more
    power on the left. For motor imagery:
      - T2 (right-hand) should yield negative LI in mu/beta (ERD on left,
        so left has *less* power, so right minus left is positive — actually
        positive LI for right-hand. Sign flips with handedness/dominance).
      - T1 (left-hand) symmetrically opposite, EXCEPT in Chunk 3 we saw
        T1 was bilateral rather than right-lateralized. LI for T1 will
        be near zero, which is itself a discriminative signal.

    **Why this feature is unusually well-motivated for this dataset.**
    Chunk 3's TFR analysis showed asymmetric lateralization — T2 lateralizes
    cleanly, T1 is bilateral. A C3-vs-C4 difference feature *directly*
    encodes the quantity that's class-discriminative in this subject's
    data. This is feature engineering informed by exploratory analysis,
    which is the methodological story worth telling in the portfolio piece.

    **On linear vs. log power for LI.** The ratio (a-b)/(a+b) is dimensionless,
    so units don't matter. But log-power changes the meaning: log(a)-log(b)
    over log(a)+log(b) is NOT a normalized log-ratio (the denominator can
    flip sign as powers cross 1.0). Using raw power keeps the [-1, +1] range
    interpretable. Default is `use_log_power=False`; the option exists
    in case downstream classification benefits from a less-bounded feature.

    Parameters
    ----------
    epochs : mne.Epochs
    bands : dict[str, (fmin, fmax)]
    pairs : list[(left_ch, right_ch)]
        Channel pairs to compute LI for. Pairs with missing channels are
        skipped silently; `pairs_used` reports what was actually computed.
    tmin, tmax : float
        Time window matching `compute_motor_bandpower` for consistency.
    use_log_power : bool
        If True, computes LI on log-power. See docstring note above.

    Returns
    -------
    X : np.ndarray, shape (n_epochs, n_pairs_used * n_bands)
    feature_names : list[str]
        E.g. ["LI_mu_C3-C4", "LI_beta_C3-C4", "LI_mu_FC3-FC4", ...].
    pairs_used : list[tuple[str, str]]
        Channel pairs actually computed (subset of requested).
    """
    if not bands:
        raise ValueError("`bands` must contain at least one band")
    if not pairs:
        raise ValueError("`pairs` must contain at least one channel pair")

    # Filter to pairs where BOTH channels exist in epochs.
    pairs_used: list[tuple[str, str]] = [
        (l, r) for (l, r) in pairs
        if l in epochs.ch_names and r in epochs.ch_names
    ]
    if not pairs_used:
        raise ValueError(
            f"No requested pairs have both channels present. "
            f"Requested: {pairs}, available: {epochs.ch_names[:10]}...")

    # We need linear power for the index formula; per-band per-channel.
    # Compute power on the full motor-region channel set once, then index.
    needed_channels = sorted({c for pair in pairs_used for c in pair})
    sub = _select_channels(epochs, needed_channels, strict=True)

    band_to_power: dict[str, np.ndarray] = {}
    for band_name, (fmin, fmax) in bands.items():
        band_to_power[band_name] = _welch_band_power(
            sub, fmin, fmax, tmin, tmax, log=use_log_power)

    n_epochs = len(sub)
    n_pairs = len(pairs_used)
    n_bands = len(bands)
    X = np.empty((n_epochs, n_pairs * n_bands), dtype=np.float64)
    feature_names: list[str] = []
    band_list = list(bands)

    for pair_idx, (left, right) in enumerate(pairs_used):
        l_idx = sub.ch_names.index(left)
        r_idx = sub.ch_names.index(right)
        for band_idx, band_name in enumerate(band_list):
            power = band_to_power[band_name]
            p_left = power[:, l_idx]
            p_right = power[:, r_idx]
            # Floor denominator to avoid division-by-zero on degenerate
            # epochs. With real EEG and log_power=False this is academic
            # (power is strictly positive), but documents the choice.
            denom = np.maximum(p_right + p_left, 1e-30)
            li = (p_right - p_left) / denom
            col = pair_idx * n_bands + band_idx
            X[:, col] = li
            feature_names.append(f"LI_{band_name}_{left}-{right}")

    return X, feature_names, pairs_used


# ---------------------------------------------------------------------------
# Feature family 3: CSP
# ---------------------------------------------------------------------------

class CSPFeatures(BaseEstimator, TransformerMixin):
    """sklearn-compatible CSP feature extractor.

    Wraps `mne.decoding.CSP` to:
      1. Enforce fit/transform separation (CSP learns from labels — it
         MUST be fit inside CV folds to avoid leakage).
      2. Set defaults appropriate for small-sample MI (Ledoit-Wolf
         shrinkage, log-variance output).
      3. Expose patterns for interpretability plots.

    Parameter notes
    ---------------
    n_components=6: yields pairs of filters (one maximizing class A
      variance, one class B). With ~22 epochs/class × 64 channels,
      more components risks overfitting. Sweep in CV.
    reg='ledoit_wolf': analytic shrinkage on per-class covariances.
      With n_channels (64) approaching n_epochs/class (~22), sample
      covariances are poorly conditioned and vanilla CSP eigendecomposition
      is unstable. Shrinkage is the standard small-sample fix
      (Lotte & Guan 2011).
    log=True, transform_into='average_power': returns log-variance per
      filter, the standard CSP feature for classification.
    norm_trace=False: redundant with shrinkage and can mix scales across
      trials. Explicit so the choice is documented.

    Patterns vs. filters
    --------------------
    Always plot `.patterns_` for neurophysiological interpretation, NOT
    `.filters_`. Patterns are the forward model (where the source projects
    on the scalp); filters are the backward model (how to weight channels
    to extract the source). See Haufe et al. (2014).
    """

    def __init__(
        self,
        n_components: int = 6,
        reg: Union[str, float, None] = "ledoit_wolf",
    ) -> None:
        self.n_components = n_components
        self.reg = reg

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSPFeatures":
        """Learn spatial filters from training epochs.

        Parameters
        ----------
        X : np.ndarray, shape (n_epochs, n_channels, n_times)
            Raw epoch data, already cropped to the imagery window.
        y : np.ndarray, shape (n_epochs,)
            Binary class labels.
        """
        if X.ndim != 3:
            raise ValueError(
                f"CSP expects 3D (epochs, channels, times); got {X.shape}")
        if len(np.unique(y)) != 2:
            raise ValueError(
                f"CSP here assumes 2 classes; got {np.unique(y)}")

        self._csp = MNE_CSP(
            n_components=self.n_components,
            reg=self.reg,
            log=True,
            norm_trace=False,
            transform_into="average_power",
        )
        self._csp.fit(X, y)

        self.classes_ = np.unique(y)
        self.n_channels_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project epochs into CSP feature space.

        Returns
        -------
        features : np.ndarray, shape (n_epochs, n_components)
        """
        if not hasattr(self, "_csp"):
            raise RuntimeError("CSPFeatures must be fit before transform")
        if X.ndim != 3:
            raise ValueError(
                f"CSP expects 3D (epochs, channels, times); got {X.shape}")
        if X.shape[1] != self.n_channels_in_:
            raise ValueError(
                f"Channel count mismatch: fit on {self.n_channels_in_}, "
                f"got {X.shape[1]}")
        return self._csp.transform(X)

    def get_feature_names(self) -> list[str]:
        return [f"CSP{i + 1}" for i in range(self.n_components)]

    @property
    def patterns_(self) -> np.ndarray:
        """Spatial patterns (forward model), shape (n_components, n_channels).

        Use these for topomap plotting, NOT `filters_`.
        """
        if not hasattr(self, "_csp"):
            raise RuntimeError("CSPFeatures must be fit before accessing patterns")
        return self._csp.patterns_[:self.n_components]

    @property
    def filters_(self) -> np.ndarray:
        """Spatial filters (backward model), shape (n_components, n_channels).

        For transforming new data (already done inside .transform()).
        Do not plot as scalp topographies.
        """
        if not hasattr(self, "_csp"):
            raise RuntimeError("CSPFeatures must be fit before accessing filters")
        return self._csp.filters_[:self.n_components]


def epochs_to_array(
    epochs: mne.Epochs,
    tmin: float = 0.5,
    tmax: float = 3.5,
    picks: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Crop epochs to imagery window and return (data, labels, channel_names).

    Defaults match `compute_motor_bandpower` for time-window consistency.
    `picks=None` keeps all channels (standard for CSP, which benefits
    from full spatial coverage to learn arbitrary filters). Setting
    `picks=SENSORIMOTOR_CHANNELS_ML` constrains CSP to motor channels,
    which is worth comparing in CV as a hyperparameter.

    Returns
    -------
    X : np.ndarray, shape (n_epochs, n_channels, n_times)
    y : np.ndarray, shape (n_epochs,)
        Integer labels from `epochs.events[:, -1]`.
    ch_names : list[str]
        Channels actually used (matches X's channel dimension).
    """
    # Copy because .crop() and .pick() are in-place; caller may need
    # full-window epochs elsewhere (e.g. for visualization plots).
    cropped = epochs.copy().crop(tmin=tmin, tmax=tmax)
    if picks is not None:
        cropped = _select_channels(cropped, picks, strict=False)
    X = cropped.get_data(copy=False)
    y = cropped.events[:, -1].astype(np.int64)
    return X, y, cropped.ch_names


# ---------------------------------------------------------------------------
# Combined feature builder
# ---------------------------------------------------------------------------

def build_ml_feature_set(
    epochs: mne.Epochs,
    feature_types: tuple[str, ...] = ("bandpower", "lateralization"),
    bp_bands: dict[str, tuple[float, float]] = MI_BANDS_ML,
    bp_channels: list[str] = SENSORIMOTOR_CHANNELS_ML,
    li_pairs: list[tuple[str, str]] = LATERALIZATION_PAIRS,
    tmin: float = 0.5,
    tmax: float = 3.5,
) -> MLFeatureSet:
    """Build a combined stateless feature matrix from one or more families.

    Note this builder handles only the STATELESS families (band power and
    lateralization indices). CSP is excluded by design — CSP learns from
    labels and must be fit inside CV folds, not pre-baked into a feature
    matrix. For pipelines that include CSP, build the stateless features
    here and assemble the full pipeline in notebook 04 using
    sklearn.pipeline.FeatureUnion or by hstacking inside the CV loop.

    Two safety properties this enforces:

    1. **No leakage by construction.** Stateless features can be computed
       once on the full dataset without leaking — they don't depend on
       labels. CSP can't go here for that reason.

    2. **Label alignment.** All sub-extractors return arrays in the same
       epoch order (epochs.events ordering). We hstack rather than risk
       any reordering shenanigans.

    Parameters
    ----------
    feature_types : tuple of {"bandpower", "lateralization"}
        Which stateless families to include.

    Returns
    -------
    fset : MLFeatureSet
        X, y, feature names, and metadata describing extraction params.
    """
    valid = {"bandpower", "lateralization"}
    unknown = set(feature_types) - valid
    if unknown:
        raise ValueError(
            f"Unknown feature_types: {unknown}. Valid for stateless "
            f"builder: {valid}. CSP must be assembled separately inside "
            f"CV — see notebook 04.")
    if not feature_types:
        raise ValueError("At least one feature_type required")

    blocks: list[np.ndarray] = []
    names: list[str] = []
    extraction_info: dict = {}

    if "bandpower" in feature_types:
        X_bp, bp_names, bp_chs_used = compute_motor_bandpower(
            epochs, bands=bp_bands, channels=bp_channels,
            tmin=tmin, tmax=tmax, log=True)
        blocks.append(X_bp)
        names.extend(bp_names)
        extraction_info["bandpower"] = {
            "bands": dict(bp_bands),
            "channels_requested": list(bp_channels),
            "channels_used": bp_chs_used,
            "n_features": X_bp.shape[1],
        }

    if "lateralization" in feature_types:
        X_li, li_names, li_pairs_used = compute_lateralization_indices(
            epochs, bands=bp_bands, pairs=li_pairs,
            tmin=tmin, tmax=tmax, use_log_power=False)
        blocks.append(X_li)
        names.extend(li_names)
        extraction_info["lateralization"] = {
            "bands": dict(bp_bands),
            "pairs_requested": list(li_pairs),
            "pairs_used": li_pairs_used,
            "n_features": X_li.shape[1],
        }

    y = epochs.events[:, -1].astype(np.int64)
    X = np.hstack(blocks)

    metadata = {
        "feature_types": tuple(feature_types),
        "tmin": tmin,
        "tmax": tmax,
        "sfreq": float(epochs.info["sfreq"]),
        "n_epochs": len(epochs),
        "per_family": extraction_info,
    }

    return MLFeatureSet(X=X, y=y, feature_names=names, metadata=metadata)