"""
Statistical testing for EEG analyses.

Permutation-based cluster tests for time-domain and time-frequency
data. The cluster approach is the field standard because it handles
the massive multiple-comparison problem in EEG (thousands of
time-by-channel or time-by-frequency points) by treating contiguous
clusters of supra-threshold activity as the unit of inference,
trading exact spatial/temporal localization for proper Type I error
control.

Reference: Maris & Oostenveld (2007), J Neurosci Methods 164, 177-190.
"""

from typing import Optional, Dict

import numpy as np
import mne
from mne.stats import (
    permutation_cluster_test,
    spatio_temporal_cluster_test,
)


def cluster_test_evoked(
    epochs: mne.Epochs,
    condition_a: str,
    condition_b: str,
    n_permutations: int = 1024,
    threshold: Optional[float] = None,
    tail: int = 0,
    n_jobs: int = 1,
    seed: int = 42,
) -> Dict:
    """
    Time-domain cluster permutation test between two conditions.

    Tests at every time point (averaged across channels, or per channel
    if you reshape) whether T1 differs from T2, forms clusters of
    adjacent significant points, and tests cluster masses against a
    null distribution from condition-label permutations.

    This implementation operates on a global field power-like reduction
    (RMS across channels per trial per timepoint) — gives a single test
    per timepoint. For per-channel localization, use the TFR version on
    each channel, or extend with channel adjacency.

    Parameters
    ----------
    epochs : mne.Epochs
        Cleaned epochs with both conditions present.
    condition_a, condition_b : str
        Condition labels matching epochs.event_id.
    n_permutations : int
        1024 is the MNE convention — enough for stable p-values down to
        ~0.001, fast enough to run interactively. Bump to 10000 for
        publication.
    threshold : float | None
        Cluster-forming threshold. None → MNE picks the F-value
        corresponding to p=0.05 given the sample size. Setting this
        manually (e.g., 6.0) gives more conservative cluster definitions
        and tends to produce sharper, smaller clusters.
    tail : int
        Direction of the test. NOTE: when MNE's default F-statistic is
        used (threshold=None or any positive threshold), this argument
        is ignored and a one-tailed F-test is performed — which is
        appropriate because the F-statistic is intrinsically
        non-negative (it tests for any difference between groups
        regardless of sign). The tail parameter is kept in the signature
        for future use with custom (e.g., t-statistic) thresholds, where
        sign matters.
    n_jobs : int
        Parallel jobs for permutations.
    seed : int
        For reproducibility of the permutation distribution.

    Returns
    -------
    results : dict
        Keys:
        - 't_obs' : ndarray, observed statistic at each timepoint
        - 'clusters' : list of cluster index arrays
        - 'cluster_p_values' : ndarray of p-values per cluster
        - 'h0' : ndarray, null distribution of max cluster stats
        - 'times' : ndarray of timepoints (seconds) for plotting

    Notes
    -----
    MNE's permutation_cluster_test expects a list of arrays, one per
    condition, each of shape (n_trials, n_features). We use
    (n_trials, n_times) by averaging the absolute value across channels
    — a GFP-like reduction. For richer spatial inference, use
    spatio_temporal_cluster_test with channel adjacency (out of scope
    for the chunk 3 first pass; revisit in chunk 6 group analysis).
    """
    if condition_a not in epochs.event_id or condition_b not in epochs.event_id:
        raise ValueError(
            f"Conditions must be in epochs.event_id "
            f"(got {condition_a!r}, {condition_b!r}; "
            f"available: {list(epochs.event_id.keys())})"
        )

    # Extract (n_trials, n_channels, n_times) data, then reduce channels
    # via root-mean-square → (n_trials, n_times). RMS is a non-negative
    # summary of "how much is happening" at each timepoint across the
    # scalp — sensitive to lateralized differences in either polarity.
    data_a = epochs[condition_a].get_data(copy=True)
    data_b = epochs[condition_b].get_data(copy=True)

    rms_a = np.sqrt((data_a ** 2).mean(axis=1))  # (n_trials_a, n_times)
    rms_b = np.sqrt((data_b ** 2).mean(axis=1))  # (n_trials_b, n_times)

    t_obs, clusters, cluster_p_values, h0 = permutation_cluster_test(
        [rms_a, rms_b],
        n_permutations=n_permutations,
        threshold=threshold,
        tail=tail,
        n_jobs=n_jobs,
        seed=seed,
        out_type='indices',
        verbose=False,
    )

    return {
        't_obs': t_obs,
        'clusters': clusters,
        'cluster_p_values': cluster_p_values,
        'h0': h0,
        'times': epochs.times,
    }


def cluster_test_tfr(
    epochs_tfr_a,
    epochs_tfr_b,
    n_permutations: int = 1024,
    threshold: Optional[float] = None,
    tail: int = 0,
    n_jobs: int = 1,
    seed: int = 42,
) -> Dict:
    """
    Time-frequency cluster permutation test.

    Same logic as cluster_test_evoked but over (time × frequency)
    instead of (time × channel). Typically run per-channel (e.g., on
    C3 and C4 separately) to test where in TF-space the conditions
    differ. Running across (time × frequency × channel) simultaneously
    is possible but requires defining channel adjacency and rarely
    adds interpretive value over per-channel tests for motor imagery.

    Parameters
    ----------
    epochs_tfr_a, epochs_tfr_b : EpochsTFR
        Per-trial TFRs from compute_tfr_morlet(..., average=False).
        Should be already restricted to the channel(s) of interest
        (typically a single channel — call .pick([ch]) beforehand).
        Baseline correction is optional and does not change the
        test's validity (it's a linear transformation that cancels
        in the difference), but for interpretation we recommend
        passing baseline-corrected TFRs.
    n_permutations : int
        See cluster_test_evoked.
    threshold : float | None
        See cluster_test_evoked. None → F-statistic threshold at
        p=0.05.
    tail : int
        See cluster_test_evoked — ignored under the default F-test
        threshold, retained for future custom-statistic use.
    n_jobs : int
    seed : int

    Returns
    -------
    results : dict
        Keys:
        - 't_obs' : ndarray of shape (n_freqs, n_times)
        - 'clusters' : list of cluster boolean masks
        - 'cluster_p_values' : ndarray of p-values per cluster
        - 'h0' : null distribution
        - 'freqs', 'times' : axis labels for plotting
        - 'channel' : the channel tested (if single)
    """
    # Pull out the data: EpochsTFR.data has shape
    # (n_epochs, n_channels, n_freqs, n_times)
    data_a = epochs_tfr_a.data
    data_b = epochs_tfr_b.data

    if data_a.shape[1] != 1 or data_b.shape[1] != 1:
        raise ValueError(
            f"cluster_test_tfr expects single-channel EpochsTFR — "
            f"got {data_a.shape[1]} and {data_b.shape[1]} channels. "
            f"Call epochs_tfr.copy().pick(['CHANNEL']) before passing."
        )

    # Drop the channel dimension → (n_epochs, n_freqs, n_times). MNE's
    # permutation_cluster_test will then find clusters in 2D (freq×time)
    # with built-in adjacency (each cell connected to its 4 neighbors).
    data_a_2d = data_a[:, 0, :, :]
    data_b_2d = data_b[:, 0, :, :]

    t_obs, clusters, cluster_p_values, h0 = permutation_cluster_test(
        [data_a_2d, data_b_2d],
        n_permutations=n_permutations,
        threshold=threshold,
        tail=tail,
        n_jobs=n_jobs,
        seed=seed,
        out_type='mask',  # boolean masks easier to plot as overlays
        verbose=False,
    )

    return {
        't_obs': t_obs,
        'clusters': clusters,
        'cluster_p_values': cluster_p_values,
        'h0': h0,
        'freqs': epochs_tfr_a.freqs,
        'times': epochs_tfr_a.times,
        'channel': epochs_tfr_a.ch_names[0],
    }


def summarize_clusters(results: Dict, alpha: float = 0.05) -> str:
    """
    Human-readable summary of cluster test results.

    Useful for printing in notebooks instead of a raw dict dump.

    Parameters
    ----------
    results : dict
        Output of cluster_test_evoked or cluster_test_tfr.
    alpha : float
        Significance threshold. Default 0.05.

    Returns
    -------
    summary : str
        Multi-line summary suitable for print().
    """
    cluster_ps = results['cluster_p_values']
    n_total = len(cluster_ps)
    n_sig = int(np.sum(cluster_ps < alpha))
    n_marginal = int(np.sum((cluster_ps >= alpha) & (cluster_ps < 0.1)))

    lines = [
        f"Cluster test summary",
        f"  Total clusters found: {n_total}",
        f"  Significant at p < {alpha}: {n_sig}",
        f"  Marginal (0.05 ≤ p < 0.10): {n_marginal}",
    ]

    if n_total > 0:
        # Report all clusters in order of significance
        sort_idx = np.argsort(cluster_ps)
        lines.append("  Per-cluster p-values (sorted):")
        for rank, idx in enumerate(sort_idx[:5]):  # top 5
            mark = " *" if cluster_ps[idx] < alpha else ""
            lines.append(f"    [{rank+1}] cluster #{idx}: p = "
                         f"{cluster_ps[idx]:.4f}{mark}")

    if 'channel' in results:
        lines.insert(1, f"  Channel: {results['channel']}")

    return "\n".join(lines)