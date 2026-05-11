"""
Visualization functions for EEG analyses.

Each function takes a computed object (Evoked, AverageTFR, etc.) and
returns a matplotlib Figure. Functions do not compute — they only
arrange already-computed quantities into plots. This separation makes
the plots trivially reusable for the Streamlit dashboard later.

Design conventions:
- Every function returns the Figure object so the caller can save it,
  embed it, or further customize it.
- No plt.show() inside functions — the notebook controls display timing.
- Default sizes target portfolio readability (figsize ~ 10x6) not paper
  submission; override via mpl rcParams if you need different.
"""

from typing import Optional, List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import mne


def plot_erp_butterfly(
    evoked: mne.Evoked,
    title: Optional[str] = None,
    spatial_colors: bool = True,
) -> plt.Figure:
    """
    Butterfly plot of all channels overlaid, with a topomap inset.

    Good first look: shows overall ERP morphology, when the signal
    departs from baseline, and whether any channel is wildly different
    from the rest (a residual artifact AutoReject missed).

    Parameters
    ----------
    evoked : mne.Evoked
        From compute_evoked().
    title : str | None
        Plot title. If None, uses evoked.comment.
    spatial_colors : bool
        If True, color channels by scalp position (lateral = warm,
        midline = cool) so spatial clusters are visually obvious in
        the butterfly. Highly recommended for a first look.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # MNE's evoked.plot() with spatial_colors gives the topomap inset
    # automatically and colors channels by position. gfp='only' would
    # give just the global field power; we want individual traces.
    fig = evoked.plot(
        spatial_colors=spatial_colors,
        gfp=True,  # overlay global field power as a thick black line
        show=False,
    )

    if title is None:
        title = evoked.comment or 'ERP'
    fig.suptitle(title, fontsize=12)
    # MNE's evoked.plot manages its own layout; tight_layout fights with
    # the inset topomap's positioning.

    return fig


def plot_erp_comparison(
    evokeds: Dict[str, mne.Evoked],
    picks: List[str],
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Overlay condition ERPs at selected channels.

    Parameters
    ----------
    evokeds : dict
        {condition_label: mne.Evoked}. MNE's plot_compare_evokeds will
        plot one line per condition with a confidence band derived from
        per-trial variance — important because the ERP-as-mean hides
        the per-trial variability. The dict keys become the legend.
    picks : list
        Typically ['C3', 'C4'] to tell the lateralization story. One
        subplot per channel.
    title : str | None
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # plot_compare_evokeds returns a list (one fig per channel by default)
    # We want them side-by-side in a single figure for portfolio-friendly
    # layout, so we use the axes= argument with our own subplot grid.
    n_picks = len(picks)
    fig, axes = plt.subplots(1, n_picks, figsize=(6 * n_picks, 4),
                             sharey=True)
    if n_picks == 1:
        axes = [axes]

    for ax, ch in zip(axes, picks):
        # ci=0.95 gives a 95% bootstrap CI from trial variance — this is
        # what makes the visual T1-vs-T2 comparison honest rather than
        # showing only the means.
        mne.viz.plot_compare_evokeds(
            evokeds,
            picks=ch,
            axes=ax,
            ci=0.95,
            show=False,
            show_sensors=False,  # we'd rather label the channel in title
            title=ch,
            legend='upper right' if ax is axes[0] else False,
        )

    if title is not None:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    return fig


def plot_erp_topomap_series(
    evoked: mne.Evoked,
    times: List[float],
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Scalp topographies at fixed latencies.

    Parameters
    ----------
    evoked : mne.Evoked
        Typically the difference wave from compute_evoked_contrast().
    times : list of float
        Seconds post-stimulus. Default-suggested for motor imagery:
        [0.1, 0.3, 0.5, 1.0, 2.0, 3.0] — early sensory window,
        mid-latency, into the sustained imagery period.
    title : str | None
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # average=0.05 averages ±25 ms around each requested time — reduces
    # noise from a single time point, which at 160 Hz is only 6.25 ms.
    # This is standard practice for ERP topographies.
    fig = evoked.plot_topomap(
        times=times,
        average=0.05,
        show=False,
        colorbar=True,
        time_unit='s',
    )

    if title is not None:
        fig.suptitle(title, fontsize=12, y=1.02)
    # Note: no tight_layout() here — plot_topomap creates its own colorbar
    # axes and tight_layout fights with that layout engine.

    return fig


def plot_tfr_single_channel(
    tfr: mne.time_frequency.AverageTFR,
    channel: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Time-frequency heatmap for one channel.

    The bread-and-butter ERD plot: time on x, frequency on y, baseline-
    corrected power as color. For motor imagery, look for the blue
    (negative dB) blob in the 8–30 Hz range starting ~500 ms post-cue
    on the channel contralateral to the imagined hand.

    Parameters
    ----------
    tfr : AverageTFR
        Baseline-corrected TFR (call apply_tfr_baseline first).
    channel : str
        Single channel name, e.g. 'C3'.
    vmin, vmax : float | None
        Color limits in dB (assuming logratio baseline). If None,
        symmetric limits around 0 are chosen from the data —
        symmetric is important so that ERD (negative) and ERS
        (positive) are visually comparable. Suggested manual values
        for portfolio plots: vmin=-3, vmax=3 dB.
    title : str | None
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # If user doesn't set limits, derive symmetric ones from the data.
    # Symmetric is critical: an asymmetric colorbar would visually bias
    # toward whichever direction has larger magnitude, distorting the
    # ERD/ERS comparison.
    if vmin is None and vmax is None:
        ch_data = tfr.copy().pick([channel]).data
        abs_max = np.percentile(np.abs(ch_data), 98)  # robust to outliers
        vmin, vmax = -abs_max, abs_max

    # RdBu_r is the conventional choice: red = positive (ERS), blue =
    # negative (ERD). The _r reverses the colormap so red is high.
    fig = tfr.plot(
        picks=[channel],
        vlim=(vmin, vmax),
        cmap='RdBu_r',
        show=False,
        colorbar=True,
        title=title if title else f'TFR: {channel}',
    )

    # tfr.plot may return a list when given multiple picks; with a single
    # pick it returns a single Figure. Normalize for safety.
    if isinstance(fig, list):
        fig = fig[0]

    return fig


def plot_tfr_topomap_band(
    tfr: mne.time_frequency.AverageTFR,
    fmin: float,
    fmax: float,
    tmin: float,
    tmax: float,
    title: Optional[str] = None,
    vlim: Tuple[Optional[float], Optional[float]] = (None, None),
) -> plt.Figure:
    """
    Scalp topography of band-averaged power over a time window.

    The lateralization money-shot: average mu (8-13 Hz) power over
    1-3 s post-cue, plot on the scalp, and see contralateral
    suppression. Run this twice (T1 and T2) side-by-side to make
    the C3-vs-C4 story visual.

    Parameters
    ----------
    tfr : AverageTFR
        Baseline-corrected TFR.
    fmin, fmax : float
        Frequency band edges (Hz). E.g. (8, 13) for mu.
    tmin, tmax : float
        Time window (seconds, relative to event onset). E.g. (1.0, 3.0)
        for sustained-imagery window.
    title : str | None
    vlim : tuple of float
        (vmin, vmax) color limits. None entries auto-scale.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # plot_topomap on a TFR averages power over the specified (f, t)
    # window automatically. We pass a single time and a single fmin/fmax
    # by averaging beforehand. Cleanest approach: use .copy() to avoid
    # mutating the input, then crop and average.
    tfr_band = tfr.copy().crop(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)

    # Average across time and frequency to get a single value per channel
    # for the topomap. Resulting topomap shows mean band power in window.
    fig = tfr_band.plot_topomap(
        tmin=tmin,
        tmax=tmax,
        fmin=fmin,
        fmax=fmax,
        cmap='RdBu_r',
        vlim=vlim,
        show=False,
        colorbar=True,
    )

    if title is not None:
        fig.suptitle(title, fontsize=12, y=1.02)
    # Note: no tight_layout() — colorbar layout engine conflict.

    return fig


def plot_tfr_lateralization(
    tfr_left_fist: mne.time_frequency.AverageTFR,
    tfr_right_fist: mne.time_frequency.AverageTFR,
    channels: Tuple[str, str] = ('C3', 'C4'),
    vmin: float = -3.0,
    vmax: float = 3.0,
) -> plt.Figure:
    """
    2x2 grid: rows = conditions (left/right fist), columns = channels
    (C3/C4). Highlights the predicted diagonal pattern: stronger ERD
    on C4 for T1 (left fist) and on C3 for T2 (right fist).

    This is the single most important figure for the portfolio narrative.

    Parameters
    ----------
    tfr_left_fist : AverageTFR
        Baseline-corrected TFR for T1 (left fist imagery).
    tfr_right_fist : AverageTFR
        Baseline-corrected TFR for T2 (right fist imagery).
    channels : tuple of (str, str)
        (left_motor_channel, right_motor_channel). Default ('C3', 'C4').
    vmin, vmax : float
        Color limits in dB. Symmetric defaults so ERD/ERS comparable.

    Returns
    -------
    fig : matplotlib.figure.Figure
        2x2 grid of TFR heatmaps.
    """
    ch_left, ch_right = channels

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    # Row 0: T1 (left fist). Contralateral = right hemisphere = ch_right.
    # Row 1: T2 (right fist). Contralateral = left hemisphere = ch_left.
    # Highlighting the diagonal (where ERD should be strongest) helps the
    # reader read the figure correctly.
    panels = [
        # (axes_row, axes_col, tfr, channel, label, is_contralateral)
        (0, 0, tfr_left_fist, ch_left,
         f'T1 (left fist) @ {ch_left} — ipsilateral', False),
        (0, 1, tfr_left_fist, ch_right,
         f'T1 (left fist) @ {ch_right} — CONTRALATERAL', True),
        (1, 0, tfr_right_fist, ch_left,
         f'T2 (right fist) @ {ch_left} — CONTRALATERAL', True),
        (1, 1, tfr_right_fist, ch_right,
         f'T2 (right fist) @ {ch_right} — ipsilateral', False),
    ]

    last_im = None
    for row, col, tfr, ch, label, is_contra in panels:
        ax = axes[row, col]
        # Plot directly to the axis. tfr.plot with axes= disables its
        # own colorbar so we can use a single shared one.
        tfr.plot(
            picks=[ch],
            axes=ax,
            colorbar=False,
            vlim=(vmin, vmax),
            cmap='RdBu_r',
            show=False,
        )
        # Boldface for contralateral panels — these are where the
        # predicted effect should be strongest.
        ax.set_title(
            label,
            fontsize=10,
            fontweight='bold' if is_contra else 'normal',
        )
        last_im = ax.images[0] if ax.images else None

    # Shared colorbar on the right side
    if last_im is not None:
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        fig.colorbar(last_im, cax=cbar_ax, label='Power (dB vs baseline)')

    fig.suptitle(
        'Motor imagery lateralization: diagonal = predicted contralateral ERD',
        fontsize=13,
    )

    return fig