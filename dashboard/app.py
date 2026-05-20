"""
EEG Motor Imagery Classification — Streamlit dashboard.

Main entry point. Streamlit's multi-page app structure auto-discovers
.py files in pages/ — they appear in the sidebar in filename order.
This file is the landing page: project overview, headline numbers,
navigation hint.

Run from repo root with:
    streamlit run dashboard/app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

# ----- Page config -----
st.set_page_config(
    page_title="EEG Motor Imagery Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----- Asset paths -----
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"     # CV result CSVs from Chunk 6
DOCS_DIR = REPO_ROOT / "docs"
ASSETS_DIR = REPO_ROOT / "dashboard" / "assets"

# ----- Header -----
st.title("EEG Motor Imagery Classification")
st.markdown(
    "**End-to-end pipeline for binary motor imagery decoding on the "
    "PhysioNet EEGBCI dataset.** Classical machine learning baselines, "
    "two deep learning architectures, honest cross-subject validation, "
    "and spatial-filter interpretability analysis — all 109 subjects."
)

# ----- Headline numbers as KPI cards -----
st.subheader("Headline results")
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Best cross-subject result",
    "0.802",
    delta="+0.242 over classical",
    help="ShallowConvNet, balanced accuracy, GroupKFold across 109 subjects.",
)
col2.metric(
    "EEGNet cross-subject",
    "0.758",
    delta="+0.198 over classical",
    help="EEGNet (Lawhern 2018) under the same protocol.",
)
col3.metric(
    "Best classical baseline",
    "0.560",
    delta="+0.060 over chance",
    help="Lateralization indices + logistic regression. The ceiling for "
         "classical features at cross-subject scale.",
)
col4.metric(
    "Subjects analyzed",
    "109 / 109",
    help="All available PhysioNet EEGBCI subjects, no exclusions.",
)

st.markdown("---")

# ----- Two-column layout: narrative + figure -----
left, right = st.columns([1, 1])

with left:
    st.subheader("The narrative")
    st.markdown(
        """
        **Within-subject scores cluster around 0.57 across all methods.**
        At ~45 trials per subject, neither EEGNet's small architecture nor
        ShallowConvNet's larger one can overcome the sample-size ceiling.
        Classical features sit there too. This is a property of the dataset,
        not the methods.

        **Cross-subject is where deep learning's advantage shows.** With
        ~3,000 training trials per fold, both deep architectures cleanly
        exceed the classical 0.560 ceiling. ShallowConvNet wins because its
        "deep CSP" structure encodes the correct inductive bias for motor
        imagery: log band-power over a learned spatial filter.

        **The spatial-filter analysis is the most interesting finding.**
        ShallowConvNet's learned filters localize cleanly to motor cortex
        (C3, CP3, FCz, C4) — the expected anatomy. EEGNet's filters
        concentrate on frontal channels (F8, AF8, F7) — the same residual
        EMG/EOG contamination that wrecked classical CSP. So Shallow's
        4-point advantage isn't just a number; it's solving the task using
        physiologically meaningful features rather than correlated artifact.
        """
    )

with right:
    st.subheader("Method comparison")
    # Defer to summary figure if it exists; otherwise build a small table.
    summary_fig = RESULTS_DIR / "chunk6_summary.png"
    if summary_fig.exists():
        st.image(str(summary_fig), use_column_width=True)
    else:
        comparison = pd.DataFrame({
            "Method": ["Random chance", "Classical CSP + LDA",
                       "Band power + SVC", "Lateralization + LogReg",
                       "EEGNet", "ShallowConvNet"],
            "Within-subject": [0.500, 0.484, 0.574, 0.580, 0.570, 0.565],
            "Cross-subject": [0.500, 0.501, 0.555, 0.560, 0.758, 0.802],
        })
        st.dataframe(
            comparison.style.format({
                "Within-subject": "{:.3f}",
                "Cross-subject": "{:.3f}",
            }).background_gradient(
                subset=["Within-subject", "Cross-subject"],
                cmap="Greens", vmin=0.5, vmax=0.85,
            ),
            hide_index=True,
            use_container_width=True,
        )

st.markdown("---")

# ----- Navigation hint -----
st.subheader("Explore further")
st.markdown(
    """
    Use the sidebar to navigate to:

    - **📊 Results Explorer** — Per-subject score distributions, fold-level
      diagnostics, response-rate breakdowns. The data behind the headline
      numbers.
    - **🧠 Spatial Filter Topomaps** — Learned spatial filters from each
      architecture, visualized as scalp topomaps. The basis for the
      "ShallowConvNet learned the right thing" finding.
    - **🎯 Try It Yourself** — Upload an EDF file or use a sample subject
      and run the full preprocessing + classification pipeline live.
      Returns predicted class for each trial.
    """
)

st.markdown("---")
with st.expander("About this project"):
    st.markdown(
        f"""
        **Repository:** [Z-bros/EEG-MotorImagery](https://github.com/Z-bros/EEG-MotorImagery)

        **Dataset:** [PhysioNet EEGBCI](https://physionet.org/content/eegmmidb/1.0.0/)
        (Schalk et al. 2004) — 109 subjects, motor imagery and execution.

        **Stack:** MNE-Python · scikit-learn · AutoReject · PyTorch · Streamlit

        **Methodology details:** see [`docs/methodology.md`](
        https://github.com/Z-bros/EEG-MotorImagery/blob/main/docs/methodology.md)
        in the repo.

        **Reproducibility:** Notebooks 01–05 run locally on CPU; notebook 06
        runs on Kaggle ((https://www.kaggle.com/code/zidanefatuna/eegnet-for-github-com-z-bros-eeg-motorimagery) free T4 GPU). All preprocessing and CV is
        deterministic.
        """
    )
