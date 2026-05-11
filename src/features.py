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

    # Subset to condition, then average. Using .copy() because epochs[cond]
    # returns a view and we don't want any downstream mutation surprises.
    evoked = epochs[condition].copy().average(picks=picks)

    # Set a human-readable comment for downstream plotting (MNE uses this
    # in legends automatically when passed to plot_compare_evokeds).
    evoked.comment = condition

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