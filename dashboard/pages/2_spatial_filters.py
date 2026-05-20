"""
Page 2 — Spatial Filter Topomaps.

Interactive visualization of learned spatial filters from EEGNet and
ShallowConvNet's cross-subject CV. Side-by-side comparison, with the
quantitative localization breakdown.

Data sources:
- eegnet_cross_full_filters.npz  (keys fold0..fold4_filters / _acts)
- shallow_cross_full_filters.npz (same structure)
- A subject epoch file for the montage info object (loaded once, cached).

If the npz files are missing, the page degrades to "regenerate by running
Chunk 6 §5/§7" instructions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import streamlit as st

# ----- Page config -----
st.set_page_config(page_title="Spatial Filters", page_icon="🧠", layout="wide")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# ROI definitions (same as Chunk 6 §8)
MOTOR_CHANNELS = ["C3", "C4", "Cz", "FC3", "FC4", "FCz",
                  "CP3", "CP4", "CPz", "C1", "C2", "C5", "C6"]
FRONTAL_CHANNELS = ["Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8",
                    "F7", "F8", "F5", "F6", "FT7", "FT8"]
TEMPORAL_CHANNELS = ["T7", "T8", "TP7", "TP8"]

# ----- Header -----
st.title("🧠 Learned spatial filter topomaps")
st.markdown(
    "Cross-subject spatial filters from EEGNet and ShallowConvNet, "
    "aggregated across the 5 GroupKFold folds with sign alignment. "
    "Use the controls to inspect individual filters, or skip to the "
    "quantitative localization breakdown."
)


# ----- Loaders -----
@st.cache_resource
def load_montage_info() -> mne.Info | None:
    """Load any subject's epoch file to extract the standard_1005 montage."""
    candidates = list(PROCESSED_DIR.glob("sub-*_epo.fif"))
    if not candidates:
        return None
    try:
        epochs = mne.read_epochs(candidates[0], preload=False, verbose=False)
        return epochs.info
    except Exception:
        return None


@st.cache_data
def load_filters_npz(path: Path, n_folds: int = 5):
    """Restore per-fold filter dict from saved npz."""
    if not path.exists():
        return None
    data = np.load(path)
    cache = {}
    for fold in range(n_folds):
        try:
            cache[fold] = {
                "spatial_filters": data[f"fold{fold}_filters"],
                "activations": data[f"fold{fold}_acts"],
            }
        except KeyError:
            return None
    return cache


def sign_align_and_average(filters_per_fold: list) -> np.ndarray:
    """Sign-align filters across folds (anchored on fold 0), then average."""
    stacked = np.stack(filters_per_fold, axis=0)
    aligned = stacked.copy()
    anchor = stacked[0]
    for f_idx in range(1, stacked.shape[0]):
        for k in range(stacked.shape[1]):
            if np.dot(stacked[f_idx, k], anchor[k]) < 0:
                aligned[f_idx, k] *= -1
    return aligned.mean(axis=0)


def localization_pct(weights: np.ndarray, ch_names: list,
                     roi: list) -> float:
    """Return % of total |weight| concentrated in ROI."""
    w = np.abs(weights)
    total = w.sum() + 1e-12
    roi_sum = sum(w[ch_names.index(c)] for c in roi if c in ch_names)
    return roi_sum / total


# ----- Load resources -----
info = load_montage_info()
eegnet_cache = load_filters_npz(RESULTS_DIR / "eegnet_cross_full_filters.npz")
shallow_cache = load_filters_npz(RESULTS_DIR / "shallow_cross_full_filters.npz")

if info is None:
    st.error(
        "No montage info available — needs at least one `sub-*_epo.fif` "
        "file in `data/processed/`. Run Chunk 2 preprocessing first."
    )
    st.stop()

if eegnet_cache is None or shallow_cache is None:
    st.error(
        "Cross-subject filter caches not found in `results/`. "
        "Run Chunk 6 §5 (EEGNet) and §7 (Shallow) to generate them."
    )
    st.stop()

# ----- Aggregate filters across folds -----
eegnet_filters = [eegnet_cache[f]["spatial_filters"] for f in range(5)]
eegnet_avg = sign_align_and_average(eegnet_filters)
eegnet_acts = np.stack([eegnet_cache[f]["activations"]
                        for f in range(5)], axis=0).mean(axis=0)

# Shallow: RMS over temporal-filter axis before averaging
shallow_filters_aggregated = []
for f in range(5):
    raw = shallow_cache[f]["spatial_filters"]  # (40, 40, 64)
    spatial_only = np.sqrt((raw ** 2).mean(axis=1))  # (40, 64)
    shallow_filters_aggregated.append(spatial_only)
shallow_avg = sign_align_and_average(shallow_filters_aggregated)
shallow_acts = np.stack([shallow_cache[f]["activations"]
                         for f in range(5)], axis=0).mean(axis=0)

# Top-k by activation
TOP_K_MAX = 12
eegnet_topk_all = np.argsort(eegnet_acts)[::-1]
shallow_topk_all = np.argsort(shallow_acts)[::-1]

# ----- Controls -----
st.subheader("Top-K filter view")
col_l, col_r = st.columns([1, 3])
with col_l:
    top_k = st.slider("Show top-K filters", min_value=3, max_value=TOP_K_MAX,
                      value=6, step=1)
    show_eegnet = st.checkbox("Show EEGNet", value=True)
    show_shallow = st.checkbox("Show ShallowConvNet", value=True)

with col_r:
    st.markdown(
        "**EEGNet** filters use signed weights (clear dipolar topomaps). "
        "**ShallowConvNet** filters are RMS-aggregated over 40 temporal "
        "filters per spatial filter, so they show magnitude only — visually "
        "less crisp than EEGNet's but quantitatively interpretable for "
        "*where* on the scalp each filter attends. The 'top-3 channels' "
        "table below the figure is the cleanest readout."
    )

# ----- Plot top-K topomaps -----
selected_arch = []
if show_eegnet:
    selected_arch.append(("EEGNet", eegnet_avg, eegnet_acts, eegnet_topk_all,
                          "RdBu_r"))
if show_shallow:
    selected_arch.append(("ShallowConvNet", shallow_avg, shallow_acts,
                          shallow_topk_all, "Reds"))

if not selected_arch:
    st.warning("Select at least one architecture above.")
else:
    fig, axes = plt.subplots(len(selected_arch), top_k,
                              figsize=(2.6 * top_k, 3.0 * len(selected_arch)))
    if len(selected_arch) == 1:
        axes = axes[None, :]

    for row, (name, filters, acts, topk_all, cmap) in enumerate(selected_arch):
        for col in range(top_k):
            ax = axes[row, col]
            f_idx = topk_all[col]
            mne.viz.plot_topomap(
                filters[f_idx], info, axes=ax, show=False,
                cmap=cmap, sensors=True, contours=4,
            )
            ax.set_title(f"{name} f{f_idx}\nact={acts[f_idx]:.2f}",
                         fontsize=9)
    fig.suptitle(
        f"Top-{top_k} cross-subject spatial filters by training activation\n"
        "(sign-aligned and averaged across 5 GroupKFold folds)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

st.markdown("---")

# ----- Quantitative localization -----
st.subheader("Quantitative localization (the actual finding)")
st.markdown(
    "Visual topomap inspection can mislead. ShallowConvNet's RMS aggregation "
    "produces visually diffuse maps, but the *quantitative* localization "
    "shows it concentrates on motor cortex. EEGNet's visually-clean dipoles "
    "concentrate over frontal channels — the same failure mode classical "
    "CSP had in Chunk 5 §6."
)


def build_localization_table(name: str, filters: np.ndarray,
                             acts: np.ndarray, top_k: int) -> "pd.DataFrame":
    import pandas as pd
    ch_names = info["ch_names"]
    topk = np.argsort(acts)[::-1][:top_k]
    rows = []
    for f_idx in topk:
        w = np.abs(filters[f_idx])
        top3_idx = np.argsort(w)[-3:][::-1]
        top3 = [ch_names[i] for i in top3_idx]
        rows.append({
            "Filter": int(f_idx),
            "Activation": float(acts[f_idx]),
            "Motor %": 100 * localization_pct(filters[f_idx], ch_names,
                                              MOTOR_CHANNELS),
            "Frontal %": 100 * localization_pct(filters[f_idx], ch_names,
                                                FRONTAL_CHANNELS),
            "Temporal %": 100 * localization_pct(filters[f_idx], ch_names,
                                                 TEMPORAL_CHANNELS),
            "Top-3 channels": ", ".join(top3),
        })
    return pd.DataFrame(rows)


cols = st.columns(2)
for col, (name, filters, acts) in zip(cols, [
    ("EEGNet", eegnet_avg, eegnet_acts),
    ("ShallowConvNet", shallow_avg, shallow_acts),
]):
    with col:
        st.markdown(f"**{name}**")
        loc_df = build_localization_table(name, filters, acts, top_k)
        st.dataframe(
            loc_df.style.format({
                "Activation": "{:.3f}",
                "Motor %": "{:.1f}",
                "Frontal %": "{:.1f}",
                "Temporal %": "{:.1f}",
            }).background_gradient(
                subset=["Motor %"], cmap="Greens", vmin=10, vmax=30,
            ).background_gradient(
                subset=["Frontal %"], cmap="Reds", vmin=10, vmax=35,
            ),
            hide_index=True,
            use_container_width=True,
        )

st.markdown("---")
st.subheader("ROI anatomical reference")
diagram_path = REPO_ROOT / "docs" / "eeg_motor_roi_diagram.png"
if diagram_path.exists():
    st.image(str(diagram_path),
             caption="Motor ROI vs frontal/temporal contamination zones on "
                     "the PhysioNet 64-channel layout.",
             use_column_width=True)
else:
    st.info(
        "Anatomical reference diagram not found at "
        "`docs/eeg_motor_roi_diagram.png`. Regenerate from the notebook 06 "
        "diagram cell or copy from the original session."
    )
