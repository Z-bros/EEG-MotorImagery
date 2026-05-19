"""
PyTorch data + training infrastructure for Chunk 6.
 
Three pieces:
- `EEGDataset`: holds raw (X, y), per-channel normalization stats (computed at
  fit time on training data and stored as buffers), and the training-only
  augmentation pipeline (channel dropout, random crop, Gaussian noise).
- `train_one_fold`: model-agnostic training loop with early stopping on
  validation balanced accuracy. Returns the best model state.
- `TorchEEGClassifier`: sklearn-compatible estimator that wraps the whole
  thing. Lets us reuse Chunk 5's `within_subject_cv` / `cross_subject_cv`
  drivers from `multisubject.py` without writing a deep-learning-specific
  CV loop.
 
Design decisions worth surfacing
--------------------------------
- Normalization stats live on the Dataset, not on the classifier or a
  separate Normalizer transformer. This way the train Dataset and test
  Dataset share statistics via constructor injection (test Dataset takes
  pre-computed mean/std from the train Dataset). One source of truth.
- Augmentation Gaussian noise σ is computed from the *un-normalized* per-
  channel std at fit time, then we normalize, then noise is added with that
  pre-normalization std × 0.1. This preserves the augmentation's physical
  meaning (perturbations on the order of 10% of natural channel variance)
  rather than dragging it into z-score space where σ would be ~0.1 of an
  already-unit-variance signal — i.e. a much weaker perturbation than intended.
- Deterministic seeds: the classifier's `fit` reseeds torch + numpy from
  `random_state + fold_index` so CV with the driver from multisubject.py
  is fully reproducible.
- Early stopping watches balanced_accuracy on a held-out 15% of the *training*
  fold (NOT the test fold, which would leak). The held-out slice is stratified.
"""
 
from __future__ import annotations
 
import copy
import random
from dataclasses import dataclass
from typing import Optional
 
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from src.eegnetmods import build_model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
 
@dataclass
class NormStats:
    """Per-channel mean and std fit on training data. Shape (n_channels,)."""
    mean: np.ndarray
    std: np.ndarray
    raw_std: np.ndarray  # un-normalized std, used to scale noise augmentation
 
 
class EEGDataset(Dataset):
    """
    EEG epochs dataset.
 
    Parameters
    ----------
    X : np.ndarray, shape (n_epochs, n_channels, n_times)
    y : np.ndarray, shape (n_epochs,)
    norm_stats : NormStats or None
        If provided, use these mean/std for normalization (test-time path).
        If None, compute them from X (train-time path).
    augment : bool
        Enable channel dropout / random crop / Gaussian noise. Train-only.
    crop_frac : float
        Random crop length as fraction of full window. 0.9 means crop to 90%
        and zero-pad back to original length.
    channel_dropout_p : float
        Per-channel probability of being zeroed.
    noise_scale : float
        Gaussian noise σ as fraction of un-normalized channel std.
    rng : np.random.Generator or None
        For reproducible augmentation. If None, a default generator is created.
    """
 
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        norm_stats: Optional[NormStats] = None,
        augment: bool = False,
        crop_frac: float = 0.9,
        channel_dropout_p: float = 0.1,
        noise_scale: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if X.ndim != 3:
            raise ValueError(f"X must be (n_epochs, n_channels, n_times); got {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X/y length mismatch: {X.shape[0]} vs {y.shape[0]}")
 
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.augment = augment
        self.crop_frac = crop_frac
        self.channel_dropout_p = channel_dropout_p
        self.noise_scale = noise_scale
        self.rng = rng if rng is not None else np.random.default_rng()
 
        # Either fit or accept normalization stats.
        if norm_stats is None:
            # Per-channel: average over (epochs, times) -> shape (n_channels,)
            mean = self.X.mean(axis=(0, 2))
            std = self.X.std(axis=(0, 2)) + 1e-8
            self.norm_stats = NormStats(mean=mean, std=std, raw_std=std.copy())
        else:
            self.norm_stats = norm_stats
 
        self.n_channels = X.shape[1]
        self.n_times = X.shape[2]
        self.crop_len = max(1, int(round(self.n_times * self.crop_frac)))
 
    def __len__(self) -> int:
        return self.X.shape[0]
 
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]  # (C, T)
        y = self.y[idx]
 
        if self.augment:
            x = self._apply_augmentation(x)
 
        # Normalize per-channel. Broadcasting: (C, T) - (C, 1).
        x = (x - self.norm_stats.mean[:, None]) / self.norm_stats.std[:, None]
 
        # Insert the singleton channel-axis the conv layers expect: (1, C, T).
        return torch.from_numpy(x).unsqueeze(0).float(), torch.tensor(y, dtype=torch.long)
 
    def _apply_augmentation(self, x: np.ndarray) -> np.ndarray:
        """Returns a copy with augmentations applied (does not mutate self.X)."""
        x = x.copy()
 
        # Random time crop: pick a start, take crop_len samples, zero-pad
        # symmetrically back to n_times so downstream shape stays constant.
        if self.crop_len < self.n_times:
            start = int(self.rng.integers(0, self.n_times - self.crop_len + 1))
            cropped = x[:, start:start + self.crop_len]
            pad_left = (self.n_times - self.crop_len) // 2
            pad_right = self.n_times - self.crop_len - pad_left
            x = np.pad(cropped, ((0, 0), (pad_left, pad_right)),
                       mode="constant", constant_values=0.0)
 
        # Channel dropout: zero out a random subset of channels independently
        # per sample. Applied on un-normalized data so the zero is a true
        # absent-signal (post-normalization the zero would represent the
        # channel's *mean*, which is not the same thing).
        if self.channel_dropout_p > 0:
            mask = self.rng.random(self.n_channels) > self.channel_dropout_p
            x = x * mask[:, None]
 
        # Gaussian noise: σ proportional to the un-normalized channel std,
        # so the perturbation is on the order of 10% of natural channel variance.
        if self.noise_scale > 0:
            noise = self.rng.standard_normal(x.shape).astype(np.float32)
            noise *= (self.noise_scale * self.norm_stats.raw_std[:, None])
            x = x + noise
 
        return x
 
 
# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
 
def _set_seed(seed: int) -> None:
    """Set all the seeds. Note: torch.use_deterministic_algorithms is set in
    the notebook (it raises if any nondet op is encountered, so we leave that
    opt-in at the experiment level rather than forcing it from this module)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
 
 
def train_one_fold(
    model: nn.Module,
    train_ds: EEGDataset,
    val_ds: EEGDataset,
    *,
    device: str = "cuda",
    batch_size: int = 64,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 15,
    class_weights: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> dict:
    """
    Train one fold with early stopping on val balanced_accuracy.
 
    Returns
    -------
    dict with keys:
        'best_state' : best model state_dict (torch.save-able)
        'best_val_bacc' : float
        'history' : list of per-epoch dicts (train_loss, val_loss, val_bacc)
        'stopped_epoch' : int (1-indexed epoch at which we stopped)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
 
    if class_weights is not None:
        cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()
 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device == "cuda"))
 
    best_val_bacc = -np.inf
    best_state = None
    epochs_since_improve = 0
    history = []
    stopped_epoch = max_epochs
 
    for epoch in range(1, max_epochs + 1):
        # ---- train ----
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
 
        # ---- validate ----
        model.eval()
        val_losses = []
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                val_losses.append(criterion(logits, yb).item())
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_true.append(yb.cpu().numpy())
        preds = np.concatenate(all_preds)
        true = np.concatenate(all_true)
        val_bacc = balanced_accuracy_score(true, preds)
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
 
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_bacc": val_bacc})
        if verbose:
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_bacc={val_bacc:.3f}")
 
        # ---- early stopping ----
        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                stopped_epoch = epoch
                break
 
    return {
        "best_state": best_state,
        "best_val_bacc": float(best_val_bacc),
        "history": history,
        "stopped_epoch": stopped_epoch,
    }
 
 
# ---------------------------------------------------------------------------
# sklearn-compatible wrapper
# ---------------------------------------------------------------------------
 
class TorchEEGClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible deep classifier.
 
    Wraps model construction + normalization-fit + training + inference into a
    single estimator with fit/predict/predict_proba. Use directly with
    StratifiedKFold or GroupKFold; the CV drivers in multisubject.py see a
    normal estimator.
 
    Parameters
    ----------
    model_name : {'eegnet', 'shallow'}
    n_channels, n_times : int
        Required up front because the model's classifier-head dimension
        depends on them (no LazyLinear).
    model_kwargs : dict
        Extra kwargs forwarded to the model constructor (F1, D, kernel_length,
        dropout, etc.).
    batch_size, lr, max_epochs, patience : training-loop hyperparameters.
    val_frac : float
        Stratified hold-out from the training fold for early stopping.
    augment : bool
        Train-time augmentation on/off.
    device : 'cuda' or 'cpu'. If 'cuda' but unavailable, falls back to 'cpu'.
    random_state : int
        Base seed. Each fit() call adds `fold_index` (set externally via
        set_fold_index) so CV reproducibility is deterministic per fold.
    """
 
    def __init__(
        self,
        model_name: str = "eegnet",
        n_channels: int = 64,
        n_times: int = 720,
        model_kwargs: Optional[dict] = None,
        batch_size: int = 64,
        lr: float = 1e-3,
        max_epochs: int = 100,
        patience: int = 15,
        val_frac: float = 0.15,
        augment: bool = True,
        device: str = "cuda",
        random_state: int = 42,
    ) -> None:
        self.model_name = model_name
        self.n_channels = n_channels
        self.n_times = n_times
        self.model_kwargs = model_kwargs
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frac = val_frac
        self.augment = augment
        self.device = device
        self.random_state = random_state
        # Mutable fold index; set externally between CV folds if you want
        # seeded variation. Default 0.
        self._fold_index = 0
 
    def set_fold_index(self, idx: int) -> "TorchEEGClassifier":
        """Set the per-fold seed offset. Returns self for chaining."""
        self._fold_index = int(idx)
        return self
 
    def _resolve_device(self) -> str:
        if self.device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return self.device
 
    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchEEGClassifier":
        # Shape contract
        if X.ndim != 3:
            raise ValueError(f"Expected (n_epochs, n_channels, n_times); got {X.shape}")
        if X.shape[1] != self.n_channels:
            raise ValueError(f"n_channels mismatch: got {X.shape[1]}, "
                             f"classifier configured for {self.n_channels}")
        if X.shape[2] != self.n_times:
            raise ValueError(f"n_times mismatch: got {X.shape[2]}, "
                             f"classifier configured for {self.n_times}")
 
        seed = self.random_state + self._fold_index
        _set_seed(seed)
        rng = np.random.default_rng(seed)
 
        # Stratified inner train/val split for early stopping.
        # If a class has fewer than 2 samples, StratifiedShuffleSplit fails;
        # we fall back to a plain random split with a warning condition.
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes  # sklearn convention
        if len(classes) < 2 or counts.min() < 2:
            # Degenerate; should not happen with our balanced design, but
            # guard against it. No held-out val -> no early stopping signal,
            # so we just take a random 15% slice.
            idx = rng.permutation(len(y))
            n_val = max(1, int(round(self.val_frac * len(y))))
            val_idx = idx[:n_val]
            train_idx = idx[n_val:]
        else:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=self.val_frac, random_state=seed)
            train_idx, val_idx = next(splitter.split(X, y))
 
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
 
        # Datasets. Train Dataset computes its own norm stats from X_tr.
        # Val Dataset reuses those stats — critical for no-leakage normalization.
        train_ds = EEGDataset(
            X_tr, y_tr, norm_stats=None, augment=self.augment, rng=rng)
        val_ds = EEGDataset(
            X_val, y_val, norm_stats=train_ds.norm_stats, augment=False)
        self.norm_stats_ = train_ds.norm_stats  # stash for predict-time
 
        # Class weights from training fold class counts.
        cw = compute_class_weight(class_weight="balanced",
                                  classes=classes, y=y_tr)
 
        # Build model fresh each fit().
        model_kwargs = self.model_kwargs or {}
        model = build_model(self.model_name, n_channels=self.n_channels,
                            n_times=self.n_times, n_classes=len(classes),
                            **model_kwargs)
 
        device = self._resolve_device()
        result = train_one_fold(
            model, train_ds, val_ds,
            device=device,
            batch_size=self.batch_size,
            lr=self.lr,
            max_epochs=self.max_epochs,
            patience=self.patience,
            class_weights=cw,
        )
 
        # Restore best weights and stash.
        model.load_state_dict(result["best_state"])
        model.eval()
        self.model_ = model
        self.device_ = device
        self.train_result_ = {k: v for k, v in result.items() if k != "best_state"}
 
        # ----------------------------------------------------------------
        # §8 prep: snapshot spatial filters + rank them by training-data
        # activation magnitude. Done here so §8 in the notebook can read
        # clf.spatial_filters_ and clf.filter_activations_ without needing
        # to re-attach the training data.
        # ----------------------------------------------------------------
        self.spatial_filters_ = model.get_spatial_filters().numpy()
 
        spatial_layer = (getattr(model, "conv_spatial", None)
                         or getattr(model, "conv_spat", None))
        if spatial_layer is None:
            self.filter_activations_ = None
        else:
            activations = []
 
            def _hook(_module, _inp, out):
                # out: (B, n_filters, 1, T). Mean |.| over all dims but filters.
                activations.append(
                    out.detach().abs().mean(dim=(0, 2, 3)).cpu().numpy())
 
            handle = spatial_layer.register_forward_hook(_hook)
            try:
                infer_ds = EEGDataset(X_tr, y_tr,
                                      norm_stats=self.norm_stats_,
                                      augment=False)
                infer_loader = DataLoader(infer_ds,
                                          batch_size=self.batch_size,
                                          shuffle=False)
                with torch.no_grad():
                    for xb, _ in infer_loader:
                        xb = xb.to(device, non_blocking=True)
                        _ = model(xb)
            finally:
                handle.remove()
            self.filter_activations_ = np.stack(activations, axis=0).mean(axis=0)
 
        return self
 
    def _infer(self, X: np.ndarray) -> np.ndarray:
        """Return logits as ndarray, shape (n_epochs, n_classes)."""
        ds = EEGDataset(X, np.zeros(len(X), dtype=np.int64),
                        norm_stats=self.norm_stats_, augment=False)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        all_logits = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device_, non_blocking=True)
                all_logits.append(self.model_(xb).cpu().numpy())
        return np.concatenate(all_logits, axis=0)
 
    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = self._infer(X)
        return self.classes_[logits.argmax(axis=1)]
 
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._infer(X)
        # Softmax in numpy (stable form)
        z = logits - logits.max(axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / ez.sum(axis=1, keepdims=True)