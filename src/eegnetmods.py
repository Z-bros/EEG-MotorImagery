"""
PyTorch architectures for Chunk 6 (deep learning baselines).

Two models, two papers:
- EEGNet: Lawhern et al. 2018 (J. Neural Eng.). ~5K params. Depthwise-separable
  conv design: temporal filter -> depthwise spatial filter -> separable conv ->
  linear. The depthwise spatial conv is the analog of CSP and is what we'll
  inspect in notebook 06 §8 to ask: does the network learn motor-cortex filters
  or replicate the frontal contamination we saw in classical CSP?
- ShallowConvNet: Schirrmeister et al. 2017 (Hum. Brain Mapp.). ~50K params.
  Designed explicitly as a "deep CSP": temporal conv -> spatial conv ->
  squaring -> average pool -> log. The square/log block is the differentiable
  analog of CSP's log-variance feature.

Design notes
------------
- Architectures only. No training, no device juggling, no fit/predict.
  src/torch_data.py owns all of that.
- Input shape: (batch, 1, n_channels, n_times). The leading singleton channel
  axis is what makes 2D conv layout natural for EEG.
- Both models output raw logits of shape (batch, n_classes). Loss layer
  (CrossEntropyLoss) lives in the training loop, not here.
- kernel_length=64 is the canonical EEGNet default. At our 160 Hz sfreq that's
  ~400 ms of temporal receptive field, slightly shorter than the ~500 ms the
  original paper used at 128 Hz. We accept the discrepancy (decision locked
  in the Chunk 6 pre-flight).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building block: max-norm-constrained Linear / Conv2d
# ---------------------------------------------------------------------------
# EEGNet's paper constrains the L2 norm of the depthwise spatial kernel weights
# (max_norm=1.0) and the final classification layer weights (max_norm=0.25).
# PyTorch doesn't ship this as a layer flag the way Keras does, so we apply it
# as a forward-time hook. Cheap and stable.


def _apply_max_norm(weight: torch.Tensor, max_norm: float, dim: int) -> None:
    """In-place rescale of `weight` so that its L2 norm along `dim` <= max_norm."""
    with torch.no_grad():
        norm = weight.norm(p=2, dim=dim, keepdim=True).clamp_min(1e-8)
        scale = (max_norm / norm).clamp_max(1.0)
        weight.mul_(scale)


# ---------------------------------------------------------------------------
# EEGNet
# ---------------------------------------------------------------------------

class EEGNet(nn.Module):
    """
    EEGNet (Lawhern et al. 2018).

    Parameters
    ----------
    n_channels : int
        Number of EEG channels. Must be constant across subjects — we rely on
        Chunk 5's `_prepare_for_concat` interpolating bads instead of dropping.
    n_times : int
        Number of time samples per epoch (e.g. 720 for our 4.5 s window at 160 Hz).
    n_classes : int, default 2
        Output dimensionality. 2 for T1 vs T2.
    F1 : int, default 8
        Number of temporal filters.
    D : int, default 2
        Depth multiplier for the spatial conv. Yields F1*D spatial filters.
    F2 : int, default 16
        Number of pointwise filters in the separable conv. Canonical: F2 = F1*D.
    kernel_length : int, default 64
        Temporal kernel size in samples.
    dropout : float, default 0.5
        Dropout probability after each pool.
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times

        # ---- Block 1: temporal conv + depthwise spatial conv -----------------
        # Temporal: 2D conv with kernel (1, kernel_length). Acts as a learned
        # bandpass filter bank applied identically across channels. `padding`
        # is 'same' along the time axis to preserve length.
        self.conv_temporal = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False,
        )
        self.bn_temporal = nn.BatchNorm2d(F1)

        # Depthwise spatial: per-temporal-filter learned spatial mixing across
        # all channels. groups=F1 makes it depthwise. This is the layer whose
        # weights we'll topomap-visualize in §8.
        self.conv_spatial = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False,
        )
        self.bn_spatial = nn.BatchNorm2d(F1 * D)

        # ---- Block 2: separable conv ----------------------------------------
        # Separable = depthwise temporal conv + pointwise (1x1) mixing.
        # The depthwise temporal here uses kernel_length // 4, which gives
        # the layer a smaller temporal receptive field on already-downsampled
        # features (we average-pool by 4 between blocks).
        self.conv_sep_depthwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, kernel_length // 4),
            padding=(0, (kernel_length // 4) // 2),
            groups=F1 * D,
            bias=False,
        )
        self.conv_sep_pointwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False,
        )
        self.bn_sep = nn.BatchNorm2d(F2)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # ---- Classification head --------------------------------------------
        # Compute the post-pooling time dimension explicitly. We pool by 4
        # after block 1, then by 8 after block 2. Integer division matches
        # PyTorch's default floor behavior for AvgPool2d.
        t_after_block1 = n_times // 4
        t_after_block2 = t_after_block1 // 8
        self.classifier = nn.Linear(F2 * t_after_block2, n_classes)

        # Stash for the max-norm hook
        self._max_norm_spatial = 1.0
        self._max_norm_classifier = 0.25

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply max-norm constraints to constrained layers before the forward
        # pass. Doing it pre-forward (rather than post-step in the training
        # loop) keeps this self-contained.
        if self.training:
            _apply_max_norm(self.conv_spatial.weight, self._max_norm_spatial, dim=2)
            _apply_max_norm(self.classifier.weight, self._max_norm_classifier, dim=1)

        # x: (B, 1, C, T)
        x = self.conv_temporal(x)           # (B, F1, C, T)
        x = self.bn_temporal(x)
        x = self.conv_spatial(x)            # (B, F1*D, 1, T)
        x = self.bn_spatial(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, kernel_size=(1, 4))
        x = self.dropout1(x)

        x = self.conv_sep_depthwise(x)
        x = self.conv_sep_pointwise(x)      # (B, F2, 1, T//4)
        x = self.bn_sep(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, kernel_size=(1, 8))
        x = self.dropout2(x)

        x = x.flatten(start_dim=1)
        x = self.classifier(x)              # raw logits
        return x

    def get_spatial_filters(self) -> torch.Tensor:
        """
        Return the depthwise spatial conv weights for topomap visualization.

        Shape: (F1*D, n_channels). Each row is one learned spatial filter
        across the EEG montage — the deep-learning analog of a CSP filter.
        """
        # conv_spatial.weight shape: (F1*D, 1, n_channels, 1)
        w = self.conv_spatial.weight.detach().cpu()
        return w.squeeze(-1).squeeze(1)


# ---------------------------------------------------------------------------
# ShallowConvNet
# ---------------------------------------------------------------------------

class _Square(nn.Module):
    """Elementwise square. Lives as a module so it's visible in repr()."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class _SafeLog(nn.Module):
    """log(clamp(x, min=eps)). The clamp protects against log(0) from pooling
    over a region of all-zero post-square activations (rare but possible at
    initialization)."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x.clamp_min(self.eps))


class ShallowConvNet(nn.Module):
    """
    ShallowConvNet (Schirrmeister et al. 2017).

    Conceptually a deep CSP: temporal conv learns bandpass filters, spatial
    conv learns channel mixings (one per temporal filter), then square + mean
    pool + log compute log-variance over a sliding window.

    Defaults follow the paper (n_temporal_filters=40, temporal_kernel=25,
    spatial_kernel="all channels", pool_kernel=75, pool_stride=15).
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 2,
        n_filters_time: int = 40,
        filter_time_length: int = 25,
        n_filters_spat: int = 40,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times

        # Temporal conv. Bias deferred to the spatial conv (paper convention).
        self.conv_time = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters_time,
            kernel_size=(1, filter_time_length),
            bias=False,
        )
        # Spatial conv across all channels in one shot. Bias enabled here.
        self.conv_spat = nn.Conv2d(
            in_channels=n_filters_time,
            out_channels=n_filters_spat,
            kernel_size=(n_channels, 1),
            bias=True,
        )
        self.bn = nn.BatchNorm2d(n_filters_spat)
        self.square = _Square()
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_time_length),
                                 stride=(1, pool_time_stride))
        self.log = _SafeLog()
        self.dropout = nn.Dropout(dropout)

        # Compute flattened size for the classifier head.
        t_after_conv = n_times - filter_time_length + 1
        t_after_pool = (t_after_conv - pool_time_length) // pool_time_stride + 1
        self.classifier = nn.Linear(n_filters_spat * t_after_pool, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, C, T)
        x = self.conv_time(x)               # (B, F_t, C, T')
        x = self.conv_spat(x)               # (B, F_s, 1, T')
        x = self.bn(x)
        x = self.square(x)
        x = self.pool(x)
        x = self.log(x)
        x = self.dropout(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

    def get_spatial_filters(self) -> torch.Tensor:
        """
        Return the spatial conv weights for topomap visualization.

        Shape: (n_filters_spat, n_filters_time, n_channels). Note that unlike
        EEGNet's depthwise design, each spatial filter here is a *combination*
        of all temporal filters — visualizing single rows in isolation is less
        clean. For §8 you'll likely want to average over the temporal-filter
        dim or look at a few representative slices.
        """
        # conv_spat.weight shape: (F_s, F_t, n_channels, 1)
        return self.conv_spat.weight.detach().cpu().squeeze(-1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    name: str,
    n_channels: int,
    n_times: int,
    n_classes: int = 2,
    **kwargs,
) -> nn.Module:
    """Dispatch by name. Lets the sklearn wrapper accept a string."""
    name_lc = name.lower()
    if name_lc == "eegnet":
        return EEGNet(n_channels=n_channels, n_times=n_times,
                      n_classes=n_classes, **kwargs)
    if name_lc in ("shallow", "shallowconvnet"):
        return ShallowConvNet(n_channels=n_channels, n_times=n_times,
                              n_classes=n_classes, **kwargs)
    raise ValueError(f"Unknown model name: {name!r}")
