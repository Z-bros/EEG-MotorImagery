"""
Multi-subject orchestration for Chunk 5.

This module is the orchestration layer that sits on top of `preprocessing.py`
(Chunk 2) and `features.py` (Chunk 4). It does three jobs:

1. Run the Chunk 2 pipeline across all 109 subjects in parallel, cache the
   results to disk, and collect per-subject metadata into a single table.
2. Provide cross-validation drivers that are aware of the subject grouping:
   within-subject CV (independent per-subject runs) and cross-subject CV
   (GroupKFold by subject id).
3. Wrap the Chunk 4 feature extractors as sklearn-compatible
   transformers so the same pipeline objects can be used inside nested CV
   without leakage.

Design notes
------------
- Cache invalidation is by `PIPELINE_VERSION`. Bump the string when the
  Chunk 2 pipeline changes; stale caches become visible immediately
  (`preprocess_all` will reprocess them when `force=False` but the version
  mismatch is logged).
- Lateralization and motor-bandpower features are *fixed formulas* on
  band-limited power — they have no learned parameters, so it is safe (and
  much faster) to precompute them once per subject and reuse across CV
  folds. CSP is *trained* and MUST sit inside the CV pipeline to avoid
  leakage. The pipeline factories below preserve that asymmetry explicitly.
- Cross-subject feature scaling is non-optional. At population scale,
  bandpower magnitudes vary subject-to-subject for non-neural reasons
  (impedance, scalp/skull conductivity, electrode placement). Every
  cross-subject pipeline gets a `StandardScaler` step; within-subject
  pipelines don't need one but are given one anyway for symmetry.

Assumed imports from sibling modules (adjust if Chunk 2/4 exposed
different names):
    from .preprocessing import preprocess_subject
    from .features import (
        compute_motor_bandpower,
        compute_lateralization_indices,
        CSPFeatures,
    )
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
import mne
from joblib import Parallel, delayed
from scipy.stats import binomtest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    RepeatedStratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.multitest import multipletests

# Sibling modules. Absolute imports so this file works both when imported
# as `src.multisubject` (package context) and when joblib workers re-import
# it in fresh subprocesses (where relative imports break).
from preprocessing import preprocess_subject  # type: ignore
from features import (  # type: ignore
    compute_motor_bandpower,
    compute_lateralization_indices,
    CSPFeatures,
)

log = logging.getLogger(__name__)

# Bump this whenever the Chunk 2 pipeline changes in a way that would
# invalidate cached .fif files. Cached files whose stored version does not
# match are flagged in the returned metadata dataframe.
PIPELINE_VERSION = "v1"

# ----------------------------------------------------------------------------
# Preprocessing orchestration
# ----------------------------------------------------------------------------


@dataclass
class SubjectRecord:
    """One row of the subject-level preprocessing metadata table.

    Stored as JSON next to each subject's cached epochs so that the
    diagnostic plots in notebook 05 can be reconstructed without rerunning
    the pipeline.
    """

    subject_id: int
    pipeline_version: str
    success: bool
    n_bad_channels: int = 0
    bad_channels: list[str] = field(default_factory=list)
    n_rejected_ics: int = 0
    rejected_ics: list[int] = field(default_factory=list)
    autoreject_retention: float = float("nan")  # fraction of epochs kept
    n_epochs_t1: int = 0  # imagined left fist
    n_epochs_t2: int = 0  # imagined right fist
    runtime_seconds: float = 0.0
    error: Optional[str] = None  # populated if success=False


def _cache_paths(cache_dir: Path, subject_id: int) -> tuple[Path, Path]:
    """Return (epochs_fif_path, metadata_json_path) for one subject."""
    stem = f"sub-{subject_id:03d}"
    return cache_dir / f"{stem}_epo.fif", cache_dir / f"{stem}_meta.json"


def _process_one(
    subject_id: int,
    runs: list[int],
    cache_dir: Path,
    force: bool,
    preprocess_kwargs: dict,
) -> SubjectRecord:
    """Run Chunk 2 preprocessing for one subject. Used by joblib worker.

    Failures are caught and turned into a SubjectRecord with success=False
    rather than raised, because at N=109 we expect a handful of edge cases
    (corrupt EDFs, ICA non-convergence) and we don't want one bad subject
    to abort the whole batch.

    Diagnostics note
    ----------------
    The Chunk 2 wrapper `preprocess_subject` returns only `Epochs` (it does
    not expose the intermediate ICA object or AutoReject log). So at this
    layer we can only recover what `Epochs` itself carries:
      - epochs.info["bads"]: channels flagged but not interpolated
      - len(epochs): final epoch count after AutoReject
      - epochs.drop_log: per-original-epoch drop reasons (used to compute
        retention)
    Rejected-IC indices and the AutoReject reject log are not recoverable
    here. If they're needed for the §6 CSP-contamination diagnostic, the
    workaround is to inline the pipeline stages in this function (see
    module docstring "Diagnostics gap" note).
    """
    epo_path, meta_path = _cache_paths(cache_dir, subject_id)

    # Skip if a valid cache exists. Validity = file exists AND stored
    # pipeline version matches.
    if not force and epo_path.exists() and meta_path.exists():
        try:
            cached = json.loads(meta_path.read_text())
            if cached.get("pipeline_version") == PIPELINE_VERSION:
                return SubjectRecord(**cached)
        except (json.JSONDecodeError, TypeError):
            # Corrupt metadata — fall through and reprocess.
            log.warning("Subject %d: corrupt metadata, reprocessing", subject_id)

    t0 = time.time()
    try:
        # Chunk 2's wrapper. Signature: preprocess_subject(subject, runs,
        # save_path=None, verbose=True) -> Epochs.
        epochs = preprocess_subject(
            subject=subject_id,
            runs=runs,
            **preprocess_kwargs,
        )
        epochs.save(epo_path, overwrite=True)

        # Compute retention from the drop log. Each entry is a tuple of
        # drop reasons for one *original* event; empty tuple = kept.
        #
        # CRITICAL: drop_log contains every event found in annotations,
        # including events NOT in `event_id` (these get reason 'IGNORED').
        # For runs 4/8/12, the EDFs contain T0 (rest), T1, T2 — but
        # `epoch_data` only builds epochs for T1 and T2. The T0 events
        # show up in drop_log as 'IGNORED' and would halve the retention
        # ratio if naively counted. Exclude them so retention reflects
        # AutoReject behaviour only.
        if hasattr(epochs, "drop_log") and len(epochs.drop_log) > 0:
            relevant = [
                entry for entry in epochs.drop_log
                if "IGNORED" not in entry  # exclude events not in event_id
            ]
            n_original = len(relevant)
            n_kept = sum(1 for entry in relevant if len(entry) == 0)
            retention = n_kept / n_original if n_original else float("nan")
        else:
            retention = float("nan")

        record = SubjectRecord(
            subject_id=subject_id,
            pipeline_version=PIPELINE_VERSION,
            success=True,
            n_bad_channels=len(epochs.info.get("bads", [])),
            bad_channels=list(epochs.info.get("bads", [])),
            n_rejected_ics=0,  # not recoverable from Epochs alone
            rejected_ics=[],   # not recoverable from Epochs alone
            autoreject_retention=float(retention),
            n_epochs_t1=int((epochs.events[:, -1] == epochs.event_id["T1"]).sum()),
            n_epochs_t2=int((epochs.events[:, -1] == epochs.event_id["T2"]).sum()),
            runtime_seconds=time.time() - t0,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Subject %d preprocessing failed", subject_id)
        record = SubjectRecord(
            subject_id=subject_id,
            pipeline_version=PIPELINE_VERSION,
            success=False,
            runtime_seconds=time.time() - t0,
            error=f"{type(exc).__name__}: {exc}",
        )

    meta_path.write_text(json.dumps(asdict(record), indent=2))
    return record


def preprocess_all(
    subject_ids: Iterable[int],
    runs: list[int],
    cache_dir: str | Path,
    n_jobs: int = -1,
    force: bool = False,
    **preprocess_kwargs,
) -> pd.DataFrame:
    """Run Chunk 2 preprocessing for every subject, in parallel, with caching.

    Parameters
    ----------
    subject_ids
        Iterable of PhysioNet subject IDs (typically range(1, 110)).
    runs
        EDF run numbers; for left/right fist imagery this is [4, 8, 12].
    cache_dir
        Directory where ``sub-NNN_epo.fif`` + ``sub-NNN_meta.json`` files
        are written. Created if missing.
    n_jobs
        joblib parallelism. -1 = all cores. ICA is the bottleneck so the
        per-subject runtime is on the order of a minute; with 8 cores the
        full 109-subject run is roughly 15 minutes.
    force
        If True, reprocess every subject even if a valid cache exists.
    preprocess_kwargs
        Forwarded to :func:`preprocessing.preprocess_subject`.

    Returns
    -------
    pd.DataFrame
        One row per subject. Inspect this for the population-level
        preprocessing diagnostics in notebook 05.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    subject_ids = list(subject_ids)

    records = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_one)(sid, runs, cache_dir, force, preprocess_kwargs)
        for sid in subject_ids
    )
    return pd.DataFrame([asdict(r) for r in records])


def load_subject_epochs(subject_id: int, cache_dir: str | Path) -> mne.Epochs:
    """Load cached epochs for one subject. Raises FileNotFoundError if missing."""
    epo_path, _ = _cache_paths(Path(cache_dir), subject_id)
    if not epo_path.exists():
        raise FileNotFoundError(
            f"No cached epochs for subject {subject_id} at {epo_path}. "
            "Run preprocess_all first."
        )
    return mne.read_epochs(epo_path, preload=True, verbose="ERROR")


def load_preprocessing_metadata(cache_dir: str | Path) -> pd.DataFrame:
    """Collect all per-subject *_meta.json files into one dataframe."""
    cache_dir = Path(cache_dir)
    rows = []
    for meta_path in sorted(cache_dir.glob("sub-*_meta.json")):
        rows.append(json.loads(meta_path.read_text()))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Feature transformers — sklearn-compatible wrappers around Chunk 4 features
# ----------------------------------------------------------------------------
#
# These wrap the Chunk 4 feature functions so they can sit inside an
# sklearn Pipeline. That matters because:
#   - GridSearchCV needs a single estimator with a `param_grid` namespace.
#   - For CSP, putting it inside the pipeline is the only way to avoid
#     fitting the spatial filters on test-fold data.
#   - For Lat and Bandpower the fit() is a no-op (no learned parameters)
#     but having them as transformers means we can swap features with a
#     one-line change.
#
# The transformers operate on mne.Epochs at fit/transform time, not on
# numpy arrays. That keeps channel metadata available to the feature
# functions. The downside is that the CV splitters must split *epoch
# indices* and we slice the Epochs object ourselves — see the CV drivers
# below.


class EpochsTransformer(BaseEstimator, TransformerMixin):
    """Base class: subclasses implement _compute(epochs) -> np.ndarray (n_epochs, n_features).

    Some Chunk 4 feature functions return tuples of per-band arrays (e.g.
    (mu_feats, beta_feats, gamma_feats)). The base class doesn't enforce
    the output shape — subclasses are responsible for concatenating bands
    into the (n_epochs, n_features) matrix sklearn expects.

    Bad-channel handling
    --------------------
    Bad channels marked in `epochs.info["bads"]` are interpolated (spherical
    spline) before features are computed. Without this, subjects with motor
    channels in their bads list (e.g. subject 109's C4) cause shape
    mismatches in `compute_lateralization_indices` and
    `compute_motor_bandpower`, which expect specific channels by name.
    Interpolation reconstructs the missing channels from neighbors so
    feature dimension stays constant across subjects — critical for
    cross-subject CV where the same feature vector layout is needed for
    every subject.
    """

    def fit(self, epochs, y=None):  # noqa: D401, ARG002
        return self

    def transform(self, epochs):
        return self._compute(self._interpolate_bads(epochs))

    def _compute(self, epochs):  # pragma: no cover - abstract
        raise NotImplementedError

    @staticmethod
    def _interpolate_bads(epochs):
        """Interpolate bad channels if any. No-op if `info['bads']` is empty.

        Uses `copy=True` so the original cached Epochs object on disk is
        never modified. The interpolated copy is local to this transform
        call and discarded after feature extraction.
        """
        if epochs.info.get("bads"):
            return epochs.copy().interpolate_bads(reset_bads=True, verbose="ERROR")
        return epochs

    @staticmethod
    def _flatten_features(out):
        """Coerce whatever the Chunk 4 function returned into (n_epochs, n_features).

        Chunk 4 feature functions follow the convention:
            return X, feature_names, extra_metadata
        where only element 0 is the (n_epochs, n_features) matrix and
        elements 1+ are bookkeeping (channel pair names, band labels, etc.).
        We extract element 0 from any tuple before further processing.

        Handles four common return shapes:
          - np.ndarray of shape (n_epochs, n_features): pass through
          - np.ndarray of shape (n_epochs,): reshape to column vector
          - tuple/list where element 0 is the feature matrix: take [0]
          - pandas DataFrame: convert to ndarray
        """
        # Chunk 4's (X, names, metadata) convention: feature matrix is [0].
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, np.ndarray):
            return out if out.ndim == 2 else out.reshape(-1, 1)
        # pandas fallback (DataFrame or Series)
        if hasattr(out, "to_numpy"):
            arr = out.to_numpy()
            return arr if arr.ndim == 2 else arr.reshape(-1, 1)
        raise TypeError(
            f"Unexpected feature output type: {type(out).__name__}. "
            "Expected ndarray, tuple of (ndarray, ...), or DataFrame."
        )


class LateralizationTransformer(EpochsTransformer):
    """Wraps features.compute_lateralization_indices.

    Lateralization is (C4_power - C3_power) / (C4 + C3) in the mu and beta
    bands. Best single-subject pipeline in Chunk 4 (0.67 with LogReg).

    The Chunk 4 function returns a tuple of per-band arrays; we concatenate
    them along the feature axis to get a single (n_epochs, n_features) matrix.
    """

    def _compute(self, epochs):
        out = compute_lateralization_indices(epochs)
        return self._flatten_features(out)


class MotorBandpowerTransformer(EpochsTransformer):
    """Wraps features.compute_motor_bandpower over sensorimotor channels.

    Same tuple-of-bands return shape as lateralization; flatten the same way.
    """

    def _compute(self, epochs):
        out = compute_motor_bandpower(epochs)
        return self._flatten_features(out)


class CSPTransformer(EpochsTransformer):
    """Wraps features.CSPFeatures so it lives inside the sklearn pipeline.

    CSP is the one transformer here that is actually fit on data. Its
    spatial filters must be learned from the training fold only, hence
    the explicit ``fit`` override.

    The Chunk 4 CSPFeatures class expects a numpy array of shape
    (n_epochs, n_channels, n_times) — not an Epochs object — so we call
    ``epochs.get_data()`` before handing off. (Discovered diagnostically:
    passing Epochs directly produced ``AttributeError: 'EpochsFIF' has no
    attribute 'ndim'``.)

    Bad channels are interpolated before `get_data()` for the same reason
    as the other transformers — keeps the n_channels dimension constant
    across subjects, which CSP's covariance computation needs.
    """

    def __init__(self, n_components: int = 4):
        self.n_components = n_components

    def fit(self, epochs, y):
        epochs = self._interpolate_bads(epochs)
        self._csp = CSPFeatures(n_components=self.n_components)
        self._csp.fit(epochs.get_data(), y)
        return self

    def transform(self, epochs):
        epochs = self._interpolate_bads(epochs)
        return self._csp.transform(epochs.get_data())


# ----------------------------------------------------------------------------
# Pipeline factories
# ----------------------------------------------------------------------------


def make_lateralization_pipeline() -> Pipeline:
    return Pipeline([
        ("features", LateralizationTransformer()),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def make_bandpower_pipeline() -> Pipeline:
    return Pipeline([
        ("features", MotorBandpowerTransformer()),
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="linear", C=1.0)),
    ])


def make_csp_pipeline(n_components: int = 4) -> Pipeline:
    # CSP outputs are log-variance and roughly comparable across subjects
    # already; the scaler is included for symmetry and costs nothing.
    return Pipeline([
        ("features", CSPTransformer(n_components=n_components)),
        ("scale", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis()),
    ])


PIPELINE_FACTORIES: dict[str, Callable[[], Pipeline]] = {
    "lateralization": make_lateralization_pipeline,
    "bandpower": make_bandpower_pipeline,
    "csp": make_csp_pipeline,
}


# ----------------------------------------------------------------------------
# CV drivers
# ----------------------------------------------------------------------------


def _epochs_to_xy(epochs: mne.Epochs) -> tuple[mne.Epochs, np.ndarray]:
    """Return (epochs, y) where y is the binary label vector.

    We pass Epochs (not numpy) through the pipeline because the feature
    transformers want channel names. ``y`` is extracted from the event
    codes: T1 -> 0 (imagined left), T2 -> 1 (imagined right).
    """
    codes = epochs.events[:, -1]
    y = np.where(codes == epochs.event_id["T2"], 1, 0).astype(int)
    return epochs, y


def within_subject_cv(
    subject_ids: Iterable[int],
    pipeline_name: str,
    cache_dir: str | Path,
    n_splits: int = 5,
    n_repeats: int = 10,
    n_jobs: int = -1,
    random_state: int = 0,
    scoring: str = "balanced_accuracy",
) -> pd.DataFrame:
    """Run repeated stratified CV per subject; return one row per subject.

    Matches the Chunk 4 protocol (5-fold x 10 repeats) so per-subject
    numbers are directly comparable to subject 1's 0.67 baseline.

    Default scoring is balanced_accuracy because ~4 subjects have
    class-imbalanced epoch counts (T1/T2 ratio up to 1.5x) after
    AutoReject. Plain accuracy would let majority-class prediction
    inflate their scores; balanced_accuracy averages per-class recall and
    is the honest metric here. For balanced subjects (~95% of the
    population) balanced_accuracy and accuracy coincide.
    """
    from sklearn.metrics import get_scorer
    scorer = get_scorer(scoring)
    factory = PIPELINE_FACTORIES[pipeline_name]
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state,
    )

    def _one(sid: int) -> dict:
        epochs = load_subject_epochs(sid, cache_dir)
        X, y = _epochs_to_xy(epochs)
        scores = []
        for train_idx, test_idx in cv.split(np.zeros(len(y)), y):
            pipe = clone(factory())
            pipe.fit(X[train_idx], y[train_idx])
            scores.append(scorer(pipe, X[test_idx], y[test_idx]))
        scores = np.asarray(scores)
        return {
            "subject_id": sid,
            "pipeline": pipeline_name,
            "scoring": scoring,
            "mean": float(scores.mean()),
            "std": float(scores.std(ddof=1)),
            "ci_lo": float(np.percentile(scores, 2.5)),
            "ci_hi": float(np.percentile(scores, 97.5)),
            "n_scores": int(scores.size),
            "n_trials": int(len(y)),
        }

    rows = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_one)(sid) for sid in subject_ids
    )
    return pd.DataFrame(rows)


def _prepare_for_concat(epochs_list, target_sfreq=None, verbose="ERROR"):
    """Normalize a list of Epochs objects so mne.concatenate_epochs accepts them.

    Handles the three metadata mismatches we've seen in this dataset:
      1. Bad channels (info['bads']): interpolate per-subject using each
         subject's own bads list, then clear bads to empty.
      2. Sampling frequency (info['sfreq']): resample to target_sfreq
         (default: the MOST COMMON sfreq across subjects, not the minimum).
         This means the majority of subjects are kept at their native rate
         and only outliers are resampled. Upsampling outliers is a benign
         operation: it doesn't add information but doesn't lose any
         either, and keeps the within-subject vs cross-subject comparison
         apples-to-apples for the majority population.
      3. Channel order: reorder to match the first subject's channel list.
         PhysioNet EDFs occasionally have channels in different orders.

    This is defensive normalization — we don't know in advance which
    metadata fields Chunk 2 standardized vs. left subject-specific.
    Centralizing the fixes here keeps cross_subject_cv and nested_cv DRY.
    """
    from collections import Counter

    # Pass 1: pick target sfreq + canonical channel order
    sfreqs = [ep.info["sfreq"] for ep in epochs_list]
    if target_sfreq is None:
        # Modal rate: the rate used by the most subjects. For PhysioNet
        # this is 160 Hz (106/109 subjects); the 3 outliers at 128 Hz
        # get upsampled rather than dragging the majority down.
        target_sfreq = Counter(sfreqs).most_common(1)[0][0]
    canonical_ch_names = epochs_list[0].ch_names

    # Pass 2: apply fixes
    normalized = []
    for ep in epochs_list:
        ep = ep.copy()
        # Interpolate bad channels using this subject's own bads.
        if ep.info.get("bads"):
            ep = ep.interpolate_bads(reset_bads=True, verbose=verbose)
        ep.info["bads"] = []
        # Resample if needed.
        if ep.info["sfreq"] != target_sfreq:
            ep = ep.resample(target_sfreq, verbose=verbose)
        # Reorder channels if needed.
        if ep.ch_names != canonical_ch_names:
            ep = ep.reorder_channels(canonical_ch_names)
        normalized.append(ep)
    return normalized


def cross_subject_cv(
    subject_ids: Iterable[int],
    pipeline_name: str,
    cache_dir: str | Path,
    n_splits: int = 5,
    random_state: int = 0,
    scoring: str = "balanced_accuracy",
) -> pd.DataFrame:
    """GroupKFold by subject. Headline cross-subject experiment.

    Returns per-held-out-subject accuracies (more useful than per-fold means
    because the responder/non-responder structure carries over from
    within-subject results — you want to see which subjects EEG-decode well
    when held out vs. which collapse).

    Uses balanced_accuracy by default for the same reason as
    within_subject_cv: roughly 4 subjects have imbalanced T1/T2 counts.

    Per-subject metadata is normalized before concatenation via
    _prepare_for_concat (interpolates bads, resamples to common sfreq,
    reorders channels). This is needed because mne.concatenate_epochs
    requires identical info across all Epochs objects.
    """
    from sklearn.metrics import get_scorer
    scorer = get_scorer(scoring)
    factory = PIPELINE_FACTORIES[pipeline_name]
    subject_ids = list(subject_ids)

    # Load all subjects, then normalize and concatenate.
    all_epochs = [load_subject_epochs(sid, cache_dir) for sid in subject_ids]
    groups = []
    for sid, ep in zip(subject_ids, all_epochs):
        groups.extend([sid] * len(ep))
    all_epochs = _prepare_for_concat(all_epochs)
    epochs = mne.concatenate_epochs(all_epochs)
    _, y = _epochs_to_xy(epochs)
    groups = np.asarray(groups)

    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        gkf.split(np.zeros(len(y)), y, groups)
    ):
        pipe = clone(factory())
        pipe.fit(epochs[train_idx], y[train_idx])
        held_out = np.unique(groups[test_idx])
        for sid in held_out:
            mask = groups[test_idx] == sid
            # Need at least one example of each class for balanced_accuracy
            # to be defined; if a held-out subject has only one class
            # represented in test, fall back to plain accuracy for that row.
            y_sub = y[test_idx[mask]]
            if len(np.unique(y_sub)) < 2:
                acc = pipe.score(epochs[test_idx[mask]], y_sub)
                scoring_used = "accuracy_fallback"
            else:
                acc = scorer(pipe, epochs[test_idx[mask]], y_sub)
                scoring_used = scoring
            rows.append({
                "fold": fold_idx,
                "subject_id": int(sid),
                "pipeline": pipeline_name,
                "scoring": scoring_used,
                "accuracy": float(acc),
                "n_test_trials": int(mask.sum()),
            })
    return pd.DataFrame(rows)


def nested_cv(
    subject_ids: Iterable[int],
    pipeline_name: str,
    param_grid: dict,
    cache_dir: str | Path,
    outer_splits: int = 5,
    inner_splits: int = 3,
    n_jobs: int = -1,
    scoring: str = "balanced_accuracy",
) -> pd.DataFrame:
    """GroupKFold outer + GroupKFold inner, with GridSearchCV in the inner.

    Honest hyperparameter tuning for cross-subject generalization. The
    inner loop is still grouped by subject so the tuning step never sees
    data from the outer-held-out subjects.
    """
    from sklearn.metrics import get_scorer
    scorer = get_scorer(scoring)
    factory = PIPELINE_FACTORIES[pipeline_name]
    subject_ids = list(subject_ids)

    all_epochs = [load_subject_epochs(sid, cache_dir) for sid in subject_ids]
    groups = []
    for sid, ep in zip(subject_ids, all_epochs):
        groups.extend([sid] * len(ep))
    all_epochs = _prepare_for_concat(all_epochs)
    epochs = mne.concatenate_epochs(all_epochs)
    _, y = _epochs_to_xy(epochs)
    groups = np.asarray(groups)

    outer = GroupKFold(n_splits=outer_splits)
    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        outer.split(np.zeros(len(y)), y, groups)
    ):
        inner = GroupKFold(n_splits=inner_splits)
        search = GridSearchCV(
            estimator=clone(factory()),
            param_grid=param_grid,
            cv=inner.split(np.zeros(train_idx.size), y[train_idx], groups[train_idx]),
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
        )
        search.fit(epochs[train_idx], y[train_idx])
        test_acc = scorer(search, epochs[test_idx], y[test_idx])
        rows.append({
            "fold": fold_idx,
            "pipeline": pipeline_name,
            "scoring": scoring,
            "best_params": search.best_params_,
            "inner_best_score": float(search.best_score_),
            "outer_test_acc": float(test_acc),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------


def responder_analysis(
    within_results: pd.DataFrame, alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-subject binomial test against chance with Holm correction.

    A subject is a "responder" if their CV accuracy is significantly above
    0.5 after correcting for 109 simultaneous tests. With ~45 trials per
    subject, the binomial test on the rounded count of correct
    classifications has well-defined behaviour even when the parametric
    CV-mean CI is sloppy.
    """
    out = within_results.copy()
    n_trials = out["n_trials"].to_numpy()
    n_correct = np.round(out["mean"].to_numpy() * n_trials).astype(int)
    pvals = np.array([
        binomtest(k, n, p=0.5, alternative="greater").pvalue
        for k, n in zip(n_correct, n_trials)
    ])
    reject, pvals_holm, *_ = multipletests(pvals, alpha=alpha, method="holm")
    out["p_raw"] = pvals
    out["p_holm"] = pvals_holm
    out["responder"] = reject
    return out


def average_csp_filters(
    subject_ids: Iterable[int],
    cache_dir: str | Path,
    n_components: int = 4,
) -> np.ndarray:
    """Fit CSP on each subject's full data; return mean filters across subjects.

    Used in notebook 05 to ask: does the Chunk 4 Fp1/F8/FT7 contamination
    on subject 1's CSP1 reproduce at population level? If yes, that
    motivates the tightened-ICA ablation parked from Chunk 4.

    Per-subject preprocessing matches CSPTransformer.fit:
      - Interpolate bad channels (using each subject's own bads list)
      - Pass the raw ndarray (n_epochs, n_channels, n_times) to
        CSPFeatures.fit, not the Epochs object directly. The Chunk 4
        class doesn't know what to do with Epochs and errors out with
        ``AttributeError: 'EpochsFIF' has no attribute 'ndim'``.

    Returns
    -------
    np.ndarray
        Shape (n_components, n_channels). Channel order matches the first
        subject's Epochs object — caller is responsible for plotting with
        the right info.
    """
    filters = []
    for sid in subject_ids:
        epochs = load_subject_epochs(sid, cache_dir)
        if epochs.info.get("bads"):
            epochs = epochs.copy().interpolate_bads(
                reset_bads=True, verbose="ERROR"
            )
        _, y = _epochs_to_xy(epochs)
        csp = CSPFeatures(n_components=n_components)
        csp.fit(epochs.get_data(), y)
        # CSPFeatures is assumed to expose `.filters_` shape (n_components, n_ch);
        # if Chunk 4 named it differently (e.g. .csp_.filters_), adjust here.
        filters.append(csp.filters_)
    return np.mean(np.stack(filters), axis=0)