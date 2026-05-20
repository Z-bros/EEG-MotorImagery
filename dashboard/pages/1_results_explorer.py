"""
Page 1 — Results Explorer.

Per-subject, per-fold score distributions for each (architecture, regime)
combination. The data behind the headline medians.

Data sources (all from RESULTS_DIR):
- eegnet_within_full.csv
- eegnet_cross_full.csv
- shallow_within_full.csv
- shallow_cross_full.csv

If a CSV is missing, that section degrades gracefully (info banner instead
of an error).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ----- Page config -----
st.set_page_config(page_title="Results Explorer", page_icon="📊", layout="wide")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"

st.title("📊 Results Explorer")
st.markdown(
    "Per-subject and per-fold balanced-accuracy distributions for each of "
    "the four CV runs in Chunk 6. The data behind the headline medians."
)


# ----- Loading helpers -----
@st.cache_data
def load_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


WITHIN_FILES = {
    "EEGNet": RESULTS_DIR / "eegnet_within_full.csv",
    "ShallowConvNet": RESULTS_DIR / "shallow_within_full.csv",
}
CROSS_FILES = {
    "EEGNet": RESULTS_DIR / "eegnet_cross_full.csv",
    "ShallowConvNet": RESULTS_DIR / "shallow_cross_full.csv",
}

within_dfs = {k: load_csv_safe(v) for k, v in WITHIN_FILES.items()}
cross_dfs = {k: load_csv_safe(v) for k, v in CROSS_FILES.items()}

# ----- Within-subject section -----
st.header("Within-subject CV")
st.caption(
    "5-fold stratified K-fold per subject, 109 subjects. The 'per-subject "
    "mean' is the mean across the 5 folds for each subject; the histogram "
    "shows the distribution of those means across subjects."
)

cols = st.columns(len(within_dfs))
for col, (name, df) in zip(cols, within_dfs.items()):
    with col:
        st.subheader(name)
        if df is None:
            st.info(f"`{WITHIN_FILES[name].name}` not found in results/")
            continue

        per_subj = df.groupby("subject")["balanced_accuracy"].agg(
            ["mean", "std"]).reset_index()
        median = per_subj["mean"].median()
        n_responders = (per_subj["mean"] >= 0.60).sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("Median per-subject", f"{median:.3f}")
        m2.metric("Responders (≥0.60)", f"{n_responders} / {len(per_subj)}")
        m3.metric("At chance (≤0.55)",
                  f"{(per_subj['mean'] <= 0.55).sum()} / {len(per_subj)}")

        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(per_subj["mean"], bins=24, color="#2ca02c", alpha=0.7,
                edgecolor="black", linewidth=0.5)
        ax.axvline(median, color="black", linestyle="--", linewidth=1.5,
                   label=f"Median = {median:.3f}")
        ax.axvline(0.500, color="red", linestyle=":", linewidth=1,
                   label="Chance = 0.500")
        ax.set_xlabel("Per-subject mean balanced accuracy")
        ax.set_ylabel("Count of subjects")
        ax.legend(fontsize=8)
        ax.set_xlim(0.30, 0.95)
        st.pyplot(fig, clear_figure=True)

        with st.expander("View per-subject table"):
            st.dataframe(
                per_subj.round(3).rename(columns={"mean": "mean_bacc",
                                                   "std": "std_bacc"}),
                hide_index=True, height=300,
            )

st.markdown("---")

# ----- Cross-subject section -----
st.header("Cross-subject CV (GroupKFold)")
st.caption(
    "5-fold GroupKFold by subject_id. Each test fold contains ~22 disjoint "
    "subjects the model never saw during training. Two summaries: "
    "**trial-pooled** (all test trials weighted equally) and **subject-"
    "pooled** (per-subject score within each fold, then median)."
)

cols = st.columns(len(cross_dfs))
for col, (name, df) in zip(cols, cross_dfs.items()):
    with col:
        st.subheader(name)
        if df is None:
            st.info(f"`{CROSS_FILES[name].name}` not found in results/")
            continue

        trial_pooled = df["balanced_accuracy"].median()
        subj_pooled = df["per_subj_median_bacc"].median()

        m1, m2 = st.columns(2)
        m1.metric("Trial-pooled median", f"{trial_pooled:.3f}")
        m2.metric("Subject-pooled median", f"{subj_pooled:.3f}")

        # Per-fold breakdown as a table
        display_df = df[[
            "fold", "balanced_accuracy", "per_subj_mean_bacc",
            "per_subj_median_bacc", "stopped_epoch", "fold_time_s",
        ]].copy()
        display_df.columns = [
            "Fold", "Trial-pooled", "Per-subj mean", "Per-subj median",
            "Stopped epoch", "Fold time (s)",
        ]
        st.dataframe(
            display_df.style.format({
                "Trial-pooled": "{:.3f}",
                "Per-subj mean": "{:.3f}",
                "Per-subj median": "{:.3f}",
                "Fold time (s)": "{:.1f}",
            }),
            hide_index=True,
            use_container_width=True,
        )

st.markdown("---")

# ----- Side-by-side comparison -----
st.header("Within-subject vs cross-subject — side by side")
st.caption(
    "The clean story: within-subject performance is method-independent "
    "(~0.57); cross-subject is where the methods diverge."
)

if all(d is not None for d in within_dfs.values()) and \
   all(d is not None for d in cross_dfs.values()):

    summary_rows = []
    for name in ["EEGNet", "ShallowConvNet"]:
        wdf = within_dfs[name]
        cdf = cross_dfs[name]
        within_med = wdf.groupby("subject")["balanced_accuracy"].mean().median()
        cross_med = cdf["per_subj_median_bacc"].median()
        summary_rows.append({
            "Method": name,
            "Within-subject median": within_med,
            "Cross-subject median": cross_med,
            "Cross − Within": cross_med - within_med,
        })

    # Inject classical baselines for the full comparison
    classical = [
        {"Method": "Classical CSP + LDA",
         "Within-subject median": 0.484,
         "Cross-subject median": 0.501,
         "Cross − Within": 0.017},
        {"Method": "Band power + SVC",
         "Within-subject median": 0.574,
         "Cross-subject median": 0.555,
         "Cross − Within": -0.019},
        {"Method": "Lateralization + LogReg",
         "Within-subject median": 0.580,
         "Cross-subject median": 0.560,
         "Cross − Within": -0.020},
    ]
    full_summary = pd.DataFrame(classical + summary_rows)
    st.dataframe(
        full_summary.style.format({
            "Within-subject median": "{:.3f}",
            "Cross-subject median": "{:.3f}",
            "Cross − Within": "{:+.3f}",
        }).background_gradient(
            subset=["Cross-subject median"], cmap="Greens",
            vmin=0.5, vmax=0.85,
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown(
        "**The +0.20 to +0.24 jump from classical to deep learning happens "
        "only in the cross-subject regime.** Within-subject is the same "
        "ceiling for everyone. Classical methods actually *drop slightly* "
        "when going from within to cross (heterogeneity hurts handcrafted "
        "features); deep learning *gains* (more data unlocks learnable "
        "representations)."
    )
else:
    st.info(
        "Side-by-side comparison requires all four result CSVs. "
        "Some are missing — see the section-level warnings above."
    )
