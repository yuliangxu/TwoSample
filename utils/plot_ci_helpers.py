import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import utils.DRE_baloss as dre
from sklearn.decomposition import PCA
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import ListedColormap, BoundaryNorm


def ci_sign_category(lo, hi):
    """
    Return integer labels:
      0 = CI completely below 0
      1 = CI covers 0
      2 = CI completely above 0
    """
    lo = np.asarray(lo).ravel()
    hi = np.asarray(hi).ravel()

    cat = np.empty_like(lo, dtype=int)

    below = (hi < 0)
    above = (lo > 0)
    cover = ~(below | above)  # includes touching 0

    cat[below] = 0
    cat[cover] = 1
    cat[above] = 2
    return cat


def compare_methods_ci(method_keys, logw_mean, logw_lo, logw_hi,
                       by_which_method=None,
                       feature_names=None, top_k=30, pos=True,
                       legend_outside=True,
                       legend_title=None,
                       legend_y=1.02,          # how high above the axes
                       legend_fontsize=18,
                       legend_title_fontsize=20,
                       markerscale=1.6):
    # choose features by average mean across methods (or a specific method)
    if by_which_method is None:
        means_stack = np.vstack([np.asarray(logw_mean[k]["self"]).ravel() for k in method_keys])
    else:
        means_stack = np.vstack([np.asarray(logw_mean[by_which_method]["self"]).ravel()])

    C = means_stack.shape[1]
    if feature_names is None:
        feature_names = [f"{j}" for j in range(C)]

    avg_mean = np.mean(means_stack, axis=0)
    if pos:
        idx = np.argsort(avg_mean)[-top_k:][::-1]   # largest positive
    else:
        idx = np.argsort(avg_mean)[:top_k]          # most negative

    y = np.arange(len(idx))
    offsets = np.linspace(-0.25, 0.25, len(method_keys))

    # Make the figure a bit wider; give room on top for legend
    fig, ax = plt.subplots(
        figsize=(8.5, 0.42 * len(idx) + 2.8),
        constrained_layout=False
    )

    for off, key in zip(offsets, method_keys):
        mean = np.asarray(logw_mean[key]["self"]).ravel()[idx]
        lo   = np.asarray(logw_lo[key]["self"]).ravel()[idx]
        hi   = np.asarray(logw_hi[key]["self"]).ravel()[idx]
        xerr = np.vstack([mean - lo, hi - mean])
        ax.errorbar(
            mean, y + off, xerr=xerr,
            fmt='o', markersize=5, capsize=3,
            elinewidth=1.5, linewidth=1.5,
            label=key
        )

    ax.axvline(0, linewidth=1.5, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([feature_names[j] for j in idx])
    ax.set_xlabel("log w (posterior mean ± 95% CI)")
    ax.invert_yaxis()

    # ---- Legend: top, outside, HORIZONTAL ----
    if legend_outside:
        handles, labels = ax.get_legend_handles_labels()

        if ax.get_legend() is not None:
            ax.get_legend().remove()

        fig.legend(
            handles, labels,
            title=legend_title,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=len(method_keys),      # ← horizontal layout
            frameon=True,
            fontsize=legend_fontsize,
            title_fontsize=legend_title_fontsize,
            markerscale=markerscale
        )

        # leave space for legend
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    else:
        ax.legend(title=legend_title)

    return fig