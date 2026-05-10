# EEG Analysis Portfolio: Motor Imagery Classification

End-to-end EEG analysis project covering preprocessing, ERP & time-frequency analysis, motor imagery classification with both classical ML and deep learning, and an interactive dashboard.

**Dataset:** [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) — 109 subjects, 64 channels, 160 Hz sampling rate.

---

## Highlights

- **Full preprocessing pipeline** with filtering, ICA-based artifact removal, and epoching
- **ERP analysis** comparing motor execution vs. motor imagery conditions
- **Time-frequency analysis** revealing event-related desynchronization (ERD) in mu and beta bands
- **Motor imagery classification** comparing classical ML (CSP + LDA/SVM) against deep learning (EEGNet)
- **Interactive Streamlit dashboard** for exploring subject-level results
- Reusable, modular code in `src/` — not just notebook scripts

## Results Snapshot

> *Final results will be populated as each notebook is completed.*

| Approach | Mean Accuracy | Subjects |
|---|---|---|
| CSP + LDA | TBD | TBD |
| CSP + SVM | TBD | TBD |
| EEGNet | TBD | TBD |

---

## Repository Structure

```
eeg-analysis-portfolio/
├── notebooks/              Story-driven analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_erp_analysis.ipynb
│   ├── 04_time_frequency.ipynb
│   ├── 05_motor_imagery_classification.ipynb
│   └── 06_dashboard_demo.ipynb
├── src/                    Reusable functions imported by notebooks
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   └── visualization.py
├── dashboard/              Streamlit app
│   └── app.py
├── results/                Saved figures and trained models
├── docs/                   Methodology write-up
└── data/                   Raw and processed EEG (gitignored)
```

---

## Setup

### Option 1: conda (recommended)
```bash
conda env create -f environment.yml
conda activate eeg-portfolio
```

### Option 2: pip
```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run notebooks
```bash
jupyter lab
```

### Run the dashboard
```bash
streamlit run dashboard/app.py
```

---

## Methodology

See [`docs/methodology.md`](docs/methodology.md) for a full write-up of design decisions, preprocessing parameters, and modeling choices.

## Notebook Walkthrough

1. **Data Exploration** — Load raw data, inspect channel layout, visualize raw signals, understand event structure.
2. **Preprocessing** — Bandpass filtering, ICA artifact removal, epoching around motor events.
3. **ERP Analysis** — Compare evoked responses across conditions; identify motor-related components.
4. **Time-Frequency Analysis** — Morlet wavelets to reveal mu (8–13 Hz) and beta (13–30 Hz) ERD/ERS.
5. **Classification** — CSP feature extraction → LDA/SVM. EEGNet trained on raw epochs. Cross-subject and within-subject evaluation.
6. **Dashboard** — Interactive Streamlit app to explore any subject's results.

---

## Tech Stack

`MNE-Python` · `scikit-learn` · `PyTorch` · `NumPy` · `pandas` · `matplotlib` · `seaborn` · `Streamlit`

## License

MIT

## Author

*Your name and contact info here.*
