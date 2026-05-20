# Dashboard

Interactive Streamlit dashboard for the EEG Motor Imagery project. Three pages:

1. **Results Explorer** — per-subject score distributions and CV diagnostics
2. **Spatial Filters** — learned filter topomaps + quantitative ROI localization
3. **Try It Yourself** — live prediction on a sample subject

## Quick start

From the repo root:

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`).

## What it expects

The dashboard reads from these locations relative to the repo root:

```
results/
├── eegnet_within_full.csv             # Chunk 6 §4 output
├── eegnet_cross_full.csv              # Chunk 6 §5 output
├── shallow_within_full.csv            # Chunk 6 §6 output
├── shallow_cross_full.csv             # Chunk 6 §7 output
├── eegnet_cross_full_filters.npz      # Chunk 6 §5 filter cache
├── shallow_cross_full_filters.npz     # Chunk 6 §7 filter cache
├── chunk6_summary.png                 # Chunk 6 §9 summary figure (optional)
└── production_shallowconvnet.pt       # Final model (see below)

data/processed/
└── sub-NNN_epo.fif                    # Chunk 2 preprocessing output

docs/
└── eeg_motor_roi_diagram.png          # Anatomical reference
```

Missing files degrade gracefully — each page shows an informational banner
indicating which file is missing and how to regenerate it.

## The production model

The **Try It Yourself** page needs a single ShallowConvNet trained on all
109 subjects (no held-out fold), stored at
`results/production_shallowconvnet.pt`. Generate it by running the
`save_production_model` cell at the end of notebook 06:

```python
# Pseudocode — actual cell is in notebooks/06_deep_learning.ipynb
from src.torch_data import TorchEEGClassifier
clf = TorchEEGClassifier(model_name="shallow", n_channels=N_CHANNELS,
                          n_times=N_TIMES, augment=True,
                          device=DEVICE, random_state=42)
clf.fit(X, y)   # train on ALL data, no held-out fold

torch.save({
    "state_dict": clf.model_.state_dict(),
    "n_channels": N_CHANNELS,
    "n_times": N_TIMES,
    "norm_mean": clf.norm_stats_.mean,
    "norm_std": clf.norm_stats_.std,
    "training": {
        "architecture": "ShallowConvNet",
        "n_subjects": int(len(np.unique(groups))),
        "n_trials_total": int(len(X)),
    },
}, RESULTS_DIR / "production_shallowconvnet.pt")
```

Run it after the cross-subject CV runs are complete. Single training, ~5
minutes on a Kaggle T4.

## Deployment

For Streamlit Cloud or similar:

1. Push the repo to GitHub (already done).
2. Connect Streamlit Cloud to the repo.
3. Set the main file to `dashboard/app.py`.
4. Add `dashboard/requirements.txt` to the project dependencies.

**Caveat for free-tier hosting:** the `production_shallowconvnet.pt` file
and the cached `data/processed/*.fif` files together are ~2 GB. Free
Streamlit Cloud has a 1 GB limit. Options:

- Host the model and a few sample subjects only (subset to ~5 subjects
  for the demo).
- Use Hugging Face Spaces (higher storage limit) or a paid Streamlit tier.
- Run locally for the live demo and use a screencast for the portfolio.
