"""
src/preprocessing.py

Preprocessing pipeline for the PhysioNet EEG Motor Movement/Imagery Dataset.

Pipeline stages (in order):
    1. load_subject()           - Load EDF runs, concatenate, standardize names, montage
    2. apply_filters()          - Bandpass 1-40 Hz, notch 60/120 Hz
    3. detect_bad_channels()    - Statistical detection (optional pyprep path)
    4. set_reference()          - Average reference (after bad-channel removal)
    5. run_ica()                - Picard ICA, 20 components
    6. reject_ica_components()  - Flag eye/muscle components, optionally apply
    7. epoch_data()             - Event-locked epochs, -0.5 to 4.0 s
    8. autoreject_epochs()      - AutoReject for residual bad epochs

Top-level wrapper:
    preprocess_subject()        - Runs full pipeline, returns cleaned Epochs.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, Sequence

import numpy as np
import mne
from mne.datasets import eegbci
from mne.preprocessing import ICA
from mne.io import BaseRaw
from mne import Epochs


# ---------------------------------------------------------------------------
# Stage 1: Loading
# ---------------------------------------------------------------------------

def load_subject(
    subject: int,
    runs: Sequence[int] = (4, 8, 12),
    data_path: Optional[Path] = None,
    preload: bool = True,
) -> BaseRaw:
    """
    Load and concatenate EDF runs for one subject from the PhysioNet EEGBCI dataset.

    Standardizes channel names and applies the standard_1005 montage so
    topomaps and ICA have correct sensor positions.

    Parameters
    ----------
    subject : int
        Subject ID (1-109).
    runs : sequence of int, default (4, 8, 12)
        Run numbers. 4/8/12 = imagined left vs right fist.
    data_path : Path, optional
        Local cache directory. If None, uses MNE's default (~/mne_data).
    preload : bool, default True
        Load data into RAM. Required for filtering and ICA.

    Returns
    -------
    raw : mne.io.Raw
    """
    # Download (or locate cached) EDF paths for the requested runs
    raw_fnames = eegbci.load_data(subject, runs, path=data_path)

    # Read each EDF and concatenate into one continuous Raw object
    raws = [mne.io.read_raw_edf(f, preload=preload) for f in raw_fnames]
    raw = mne.concatenate_raws(raws)

    # Strip trailing dots from EDF channel names ('Fc5.' -> 'FC5') so they
    # match the standard_1005 montage
    eegbci.standardize(raw)

    # Apply the 10-05 montage (3D electrode positions)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage)

    return raw


# ---------------------------------------------------------------------------
# Stage 2: Filtering
# ---------------------------------------------------------------------------

def apply_filters(
    raw: BaseRaw,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    notch_freqs: Sequence[float] = (60.0,),
    fir_design: str = "firwin",
    copy: bool = True,
) -> BaseRaw:
    raw_filt = raw.copy() if copy else raw

    # Drop any notch frequencies at or above Nyquist — they're not
    # representable at this sample rate. Filter them silently rather than
    # failing, since the bandpass below 40 Hz makes them redundant anyway.
    nyquist = raw_filt.info["sfreq"] / 2.0
    valid_notch = [f for f in notch_freqs if f < nyquist]

    if valid_notch:
        raw_filt.notch_filter(
            freqs=valid_notch,
            fir_design=fir_design,
            verbose=False,
        )

    raw_filt.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design=fir_design,
        verbose=False,
    )

    return raw_filt

# ---------------------------------------------------------------------------
# Stage 3: Bad channel detection
# ---------------------------------------------------------------------------

def detect_bad_channels(
    raw: BaseRaw,
    method: str = "statistical",
    z_threshold: float = 3.0,
    flat_threshold_uv: float = 1.0,
    correlation_threshold: float = 0.4,
    mark_in_place: bool = True,
) -> list[str]:
    """
    Detect bad channels using statistical criteria or pyprep's NoisyChannels.

    Statistical method:
        - FLAT: std < flat_threshold_uv (likely disconnected)
        - HIGH VARIANCE: log-variance z-score > z_threshold
        - LOW CORRELATION: max correlation with any other channel < correlation_threshold

    pyprep method:
        Uses pyprep.find_noisy_channels.NoisyChannels. More robust, slower,
        adds dependency. Recommended for multi-subject runs.

    Parameters
    ----------
    raw : mne.io.Raw
    method : {'statistical', 'pyprep'}
    z_threshold, flat_threshold_uv, correlation_threshold : float
    mark_in_place : bool
        If True, append detected channels to raw.info['bads'].

    Returns
    -------
    bads : list of str
    """
    if method == "pyprep":
        try:
            from pyprep.find_noisy_channels import NoisyChannels
        except ImportError as e:
            raise ImportError(
                "pyprep is not installed. Run `pip install pyprep` or "
                "use method='statistical'."
            ) from e
        nd = NoisyChannels(raw.copy(), random_state=97)
        nd.find_all_bads()
        bads = nd.get_bads()

    elif method == "statistical":
        # Pick only EEG channels (skip stim/EOG if present)
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        data = raw.get_data(picks=picks)  # shape (n_channels, n_samples), volts
        ch_names = [raw.ch_names[i] for i in picks]

        bads: list[str] = []

        # --- 1. Flat channels: std below flat_threshold_uv ---
        stds = data.std(axis=1)  # per-channel std in volts
        flat_mask = stds < (flat_threshold_uv * 1e-6)  # convert µV -> V
        bads.extend([ch for ch, flat in zip(ch_names, flat_mask) if flat])

        # --- 2. High-variance channels: log-variance z-score > threshold ---
        # Log-variance is more normal than variance, so z-scoring is meaningful
        # Add tiny epsilon to avoid log(0) on flat channels (already flagged)
        log_var = np.log(stds ** 2 + 1e-30)
        z = (log_var - log_var.mean()) / log_var.std()
        noisy_mask = z > z_threshold
        bads.extend([ch for ch, n in zip(ch_names, noisy_mask) if n and ch not in bads])

        # --- 3. Low-correlation channels: poorly correlated with all others ---
        # Bridging, isolated noise, or bad contact tends to make a channel
        # uncorrelated with its neighbors (and the rest of the head)
        corr = np.corrcoef(data)
        np.fill_diagonal(corr, 0.0)  # ignore self-correlation
        max_corr = np.abs(corr).max(axis=1)
        uncorr_mask = max_corr < correlation_threshold
        bads.extend([ch for ch, u in zip(ch_names, uncorr_mask) if u and ch not in bads])

    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'statistical' or 'pyprep'.")

    if mark_in_place:
        # Merge with any pre-existing bads, avoid duplicates
        existing = set(raw.info["bads"])
        raw.info["bads"] = list(existing | set(bads))

    return bads


# ---------------------------------------------------------------------------
# Stage 4: Reference
# ---------------------------------------------------------------------------

def set_reference(
    raw: BaseRaw,
    ref_channels: Union[str, Sequence[str]] = "average",
    projection: bool = False,
    copy: bool = True,
) -> BaseRaw:
    """
    Re-reference the EEG. Default is common average reference (CAR).

    Bad channels (raw.info['bads']) are excluded from the average automatically
    by MNE.

    Parameters
    ----------
    raw : mne.io.Raw
    ref_channels : 'average' or list of channel names
    projection : bool
        If True, applied as a projector (reversible). We use False so ICA
        sees the referenced signal directly.
    copy : bool

    Returns
    -------
    raw_ref : mne.io.Raw
    """
    raw_ref = raw.copy() if copy else raw
    raw_ref.set_eeg_reference(
        ref_channels=ref_channels,
        projection=projection,
        verbose=False,
    )
    return raw_ref


# ---------------------------------------------------------------------------
# Stage 5: ICA
# ---------------------------------------------------------------------------

def run_ica(
    raw: BaseRaw,
    n_components: int = 20,
    method: str = "picard",
    random_state: int = 97,
    fit_params: Optional[dict] = None,
) -> ICA:
    """
    Fit ICA for artifact decomposition.

    Rationale:
        n_components=20: separates common artifacts (blink, saccade, muscle)
            from neural sources without overfitting.
        method='picard': fast, MNE-recommended.
        random_state=97: ICA is non-deterministic without this.

    Parameters
    ----------
    raw : mne.io.Raw
        Must be filtered (1 Hz HPF required for stable ICA).
    n_components : int or float
    method : str
    random_state : int
    fit_params : dict, optional

    Returns
    -------
    ica : mne.preprocessing.ICA
        Fitted but not applied. Caller sets ica.exclude.
    """
    ica = ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        fit_params=fit_params,
        max_iter="auto",
    )
    # Fit on EEG channels only; ICA respects raw.info['bads'] automatically
    ica.fit(raw, picks="eeg", verbose=False)
    return ica


def reject_ica_components(
    raw: BaseRaw,
    ica: ICA,
    eog_ch: Optional[str] = "Fp1",
    eog_threshold: float = 3.0,
    muscle_threshold: float = 0.5,
    apply: bool = True,
    copy: bool = True,
) -> tuple[BaseRaw, dict]:
    """
    Auto-identify and (optionally) remove eye and muscle ICA components.

    EOG detection: correlates each component with a frontal channel (Fp1
        by default) as a pseudo-EOG, since this dataset has no real EOG.
    Muscle detection: uses MNE's find_bads_muscle, which scores components
        on HF power and peripheral spatial signature.

    Parameters
    ----------
    raw : mne.io.Raw
    ica : ICA
        Already fitted.
    eog_ch : str
    eog_threshold : float
    muscle_threshold : float
    apply : bool
        If True, apply ICA exclusion and return cleaned raw.
        If False, populate ica.exclude only and return raw unchanged.
    copy : bool

    Returns
    -------
    raw_out : mne.io.Raw
    info : dict
        {'eog_components', 'muscle_components', 'eog_scores', 'muscle_scores'}
    """
    info = {
        "eog_components": [],
        "muscle_components": [],
        "eog_scores": None,
        "muscle_scores": None,
    }

    # --- EOG components via frontal-channel proxy ---
    # find_bads_eog returns (indices, scores). If eog_ch isn't in the data
    # we fall back gracefully — no EOG rejection rather than a crash.
    if eog_ch is not None and eog_ch in raw.ch_names:
        eog_inds, eog_scores = ica.find_bads_eog(
            raw, ch_name=eog_ch, threshold=eog_threshold, verbose=False,
        )
        info["eog_components"] = list(eog_inds)
        info["eog_scores"] = eog_scores

    # --- Muscle components ---
    muscle_inds, muscle_scores = ica.find_bads_muscle(
        raw, threshold=muscle_threshold, verbose=False,
    )
    info["muscle_components"] = list(muscle_inds)
    info["muscle_scores"] = muscle_scores

    # Combine into the ICA's exclude list (de-duplicated)
    ica.exclude = sorted(set(ica.exclude) | set(info["eog_components"]) | set(info["muscle_components"]))

    if apply:
        raw_out = raw.copy() if copy else raw
        ica.apply(raw_out, verbose=False)
    else:
        raw_out = raw

    return raw_out, info


# ---------------------------------------------------------------------------
# Stage 6: Epoching
# ---------------------------------------------------------------------------

def epoch_data(
    raw: BaseRaw,
    event_id: Optional[dict] = None,
    tmin: float = -0.5,
    tmax: float = 4.0,
    baseline: Optional[tuple] = (-0.5, 0.0),
    preload: bool = True,
) -> Epochs:
    """
    Extract event-locked epochs.

    Window -0.5 to 4.0 s = 720 samples at 160 Hz, covering pre-cue baseline
    plus the full 4-second imagination period.

    For runs 4/8/12: T1=left fist imagery, T2=right fist imagery.
    Rest periods (T0) are dropped.

    Parameters
    ----------
    raw : mne.io.Raw
    event_id : dict, optional
        If None, built from annotations, keeping T1 and T2.
    tmin, tmax : float
    baseline : tuple or None
    preload : bool

    Returns
    -------
    epochs : mne.Epochs
    """
    # Extract events from annotations. events_from_annotations builds a
    # default event_id mapping like {'T0': 1, 'T1': 2, 'T2': 3}.
    events, ann_event_id = mne.events_from_annotations(raw, verbose=False)

    if event_id is None:
        # Keep only T1 and T2 (the motor imagery classes); drop T0 (rest)
        event_id = {k: v for k, v in ann_event_id.items() if k in ("T1", "T2")}

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=preload,
        picks="eeg",
        reject_by_annotation=True,
        verbose=False,
    )
    return epochs


# ---------------------------------------------------------------------------
# Stage 7: AutoReject
# ---------------------------------------------------------------------------

def autoreject_epochs(
    epochs: Epochs,
    n_interpolate: Sequence[int] = (1, 4, 8),
    consensus: Optional[Sequence[float]] = None,
    cv: int = 5,
    random_state: int = 97,
    return_log: bool = True,
):
    """
    Run AutoReject (Jas et al. 2017) for final epoch-level cleaning.

    Learns per-channel peak-to-peak thresholds by CV and either interpolates
    or drops each epoch.

    Parameters
    ----------
    epochs : mne.Epochs
    n_interpolate : sequence of int
    consensus : sequence of float, optional
    cv : int
    random_state : int
    return_log : bool

    Returns
    -------
    epochs_clean : mne.Epochs
    reject_log : autoreject.RejectLog (if return_log)
    """
    try:
        from autoreject import AutoReject
    except ImportError as e:
        raise ImportError(
            "autoreject is not installed. Run `pip install autoreject`."
        ) from e

    ar = AutoReject(
        n_interpolate=np.array(n_interpolate),
        consensus=np.array(consensus) if consensus is not None else None,
        cv=cv,
        random_state=random_state,
        n_jobs=1,
        verbose=False,
    )

    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

    if return_log:
        return epochs_clean, reject_log
    return epochs_clean


# ---------------------------------------------------------------------------
# Top-level wrapper
# ---------------------------------------------------------------------------

def preprocess_subject(
    subject: int,
    runs: Sequence[int] = (4, 8, 12),
    save_path: Optional[Path] = None,
    verbose: bool = True,
) -> Epochs:
    """
    Run the full preprocessing pipeline end-to-end for one subject.

    Stages: load -> filter -> detect bads -> reference -> ICA fit ->
    ICA reject -> epoch -> autoreject.

    Parameters
    ----------
    subject : int
    runs : sequence of int
    save_path : Path, optional
        If given, saves cleaned epochs as `sub-{subject:03d}_epo.fif`.
    verbose : bool

    Returns
    -------
    epochs_clean : mne.Epochs
    """
    def _log(msg):
        if verbose:
            print(msg)

    _log(f"[sub-{subject:03d}] Loading runs {list(runs)}...")
    raw = load_subject(subject, runs=runs)

    _log(f"[sub-{subject:03d}] Filtering (1–40 Hz, notch 60/120)...")
    raw = apply_filters(raw, copy=False)

    _log(f"[sub-{subject:03d}] Detecting bad channels...")
    bads = detect_bad_channels(raw, method="statistical")
    _log(f"[sub-{subject:03d}]   bads: {bads}")

    _log(f"[sub-{subject:03d}] Applying average reference...")
    raw = set_reference(raw, copy=False)

    _log(f"[sub-{subject:03d}] Fitting ICA (20 components, Picard)...")
    ica = run_ica(raw)

    _log(f"[sub-{subject:03d}] Rejecting EOG/muscle ICA components...")
    raw, ica_info = reject_ica_components(raw, ica, apply=True, copy=False)
    _log(f"[sub-{subject:03d}]   excluded components: {ica.exclude}")

    _log(f"[sub-{subject:03d}] Epoching (-0.5 to 4.0 s)...")
    epochs = epoch_data(raw)
    _log(f"[sub-{subject:03d}]   {len(epochs)} epochs before AutoReject")

    _log(f"[sub-{subject:03d}] Running AutoReject...")
    epochs_clean, _ = autoreject_epochs(epochs)
    _log(f"[sub-{subject:03d}]   {len(epochs_clean)} epochs after AutoReject")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        fname = save_path / f"sub-{subject:03d}_epo.fif"
        epochs_clean.save(fname, overwrite=True)
        _log(f"[sub-{subject:03d}] Saved to {fname}")

    return epochs_clean