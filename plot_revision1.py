# %%
# load packages
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import utils.DRE_baloss as dre
from sklearn.decomposition import PCA
from matplotlib.colors import TwoSlopeNorm
from importlib import reload
import utils.microbiome_help as help
import utils.plot_ci_helpers as plot_ci_helpers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
# %% load all data
# data_path = "/hpc/group/mastatlab/microbiome/"
data_path = "./data/sample_"
ratio_test_data_path = "./data/revision/test/"
ratio_train_data_path = "./data/revision/train/"


sample_test = np.loadtxt(data_path+'test.csv',delimiter=',')
sample_train = np.loadtxt(data_path+'train.csv',delimiter=',')

# parametric/non parametric
sample_d = np.loadtxt(data_path+'d.csv',delimiter=',')
sample_dt = np.loadtxt(data_path+'dt.csv',delimiter=',')

# CFM
sample_icfm = np.loadtxt(data_path+'icfm.csv',delimiter=',')
# GAN
sample_mbgan = np.loadtxt(data_path+'mbgan.csv',delimiter=',')

# %% load log_w results

own_samples = {
    "d":        sample_d,
    "dt":       sample_dt,
    "icfm":   sample_icfm,
    "mbgan":    sample_mbgan,
}


# (4) Now loop over each method, run it on sample_train and on its own sample:
log_w_gradient = {}
log_w_boosting = {}
log_w_bayesian = {}
log_w_bayesian_CI_025 = {}
log_w_bayesian_CI_975 = {}
n_train = sample_train.shape[0]
n_test = sample_test.shape[0]

for key in own_samples.keys():
    result = np.loadtxt(ratio_test_data_path+"log_w_"+key+'.csv',delimiter=',')
    # If you want NumPy arrays rather than Tensors:
    log_w_test = result[:n_test,:]
    log_w_self  = result[n_test:,:]
    log_w_gradient[key] = {
        "test": log_w_test[:,0],
        "self":  log_w_self[:,0]
    }
    log_w_boosting[key] = {
        "test": log_w_test[:,1],
        "self":  log_w_self[:,1]
    }
    log_w_bayesian[key] = {
        "test": log_w_test[:,2],
        "self":  log_w_self[:,2]
    }
    log_w_bayesian_CI_025[key] = {
        "test": log_w_test[:,3],
        "self":  log_w_self[:,3]
    }
    log_w_bayesian_CI_975[key] = {
        "test": log_w_test[:,9],
        "self":  log_w_self[:,9]
    }

# 
log_w_gradient_train = {}
log_w_boosting_train = {}
log_w_bayesian_train = {}
log_w_bayesian_CI_025_train = {}
log_w_bayesian_CI_975_train = {}

for key in own_samples.keys():
    result = np.loadtxt(ratio_train_data_path+"log_w_"+key+'.csv',delimiter=',')
    # If you want NumPy arrays rather than Tensors:
    log_w_train = result[:n_train,:]
    log_w_self  = result[n_train:,:]
    log_w_gradient_train[key] = {
        "train": log_w_train[:,0],
        "self":  log_w_self[:,0]
    }
    log_w_boosting_train[key] = {
        "train": log_w_train[:,1],
        "self":  log_w_self[:,1]
    }
    log_w_bayesian_train[key] = {
        "train": log_w_train[:,2],
        "self":  log_w_self[:,2]
    }
    log_w_bayesian_CI_025_train[key] = {
        "train": log_w_train[:,3],
        "self":  log_w_self[:,3]
    }
    log_w_bayesian_CI_975_train[key] = {
        "train": log_w_train[:,9],
        "self":  log_w_self[:,9]
    }

# %% for test

log_w_results = log_w_bayesian
# log_w_results = log_w_bayesian_CI_025
# log_w_results = log_w_bayesian_CI_975

method_names = {
    "d":       "Dirichlet",
    "dt":      "Dirichlet Tree",
    "icfm":  "ICFM",
    "mbgan":   "MB-GAN"
}

from sklearn.cross_decomposition import CCA
from scipy.spatial.distance import pdist, squareform

BIG = 26   # base size; try 24–30 depending on output medium

plt.rcParams.update({
    "font.size": BIG,
    "axes.titlesize": BIG + 6,
    "axes.labelsize": BIG + 2,
    "xtick.labelsize": BIG,
    "ytick.labelsize": BIG,
    "legend.fontsize": BIG,
    "legend.title_fontsize": BIG + 2,
})
# Containers to hold all per-method data
per_method = {}
all_logw_values = []  # we'll collect every log(w) to determine global vmin/vmax
metric='braycurtis' # for PCoA plot


for key in own_samples.keys():

    # Convert compositional arrays to NumPy
    X_test  = dre.to_numpy(sample_test)        # shape: (n_test, k)
    X_self  = dre.to_numpy(own_samples[key])   # shape: (n_self, k)

    logw_test = log_w_results[key]["test"]  # shape: (n_train,)
    logw_self  = log_w_results[key]["self"]   # shape: (n_self,)

    # Collect for global vmin/vmax
    all_logw_values.append(logw_test)
    all_logw_values.append(logw_self)

    # compute CCA
    stacked_X = np.vstack([X_test,X_self])
    stacked_Y = np.vstack((logw_test[:, None],
                       logw_self[:,  None]))


    # PCoA
    pc1, pc2, pct1, pct2 = help.compute_pcoa_coords(stacked_X)
    pc1_pcoa_train = pc1[:n_test]
    pc1_pcoa_self = pc1[n_test:]
    pc2_pcoa_train = pc2[:n_test]
    pc2_pcoa_self = pc2[n_test:]


    # Save everything for this method
    per_method[key] = {
        "logw_test":     logw_test,
        "logw_self":      logw_self,
        "pc1_pcoa_train":   pc1_pcoa_train,
        "pc1_pcoa_self":    pc1_pcoa_self,
        "pc2_pcoa_train":   pc2_pcoa_train,
        "pc2_pcoa_self":    pc2_pcoa_self,
        "pct1_pcoa":    pct1,
        "pct2_pcoa":    pct2,
    }

# Stack all log(w) values to find a global range
all_logw_flat = np.concatenate(all_logw_values)
# global_vmin = np.min(all_logw_flat)
# global_vmax = np.max(all_logw_flat)
# for training
# global_vmin = -30
# global_vmax = 30

# for testing
global_vmin = -12
global_vmax = 12
cmap = plt.get_cmap('seismic')
norm = TwoSlopeNorm(vmin=global_vmin, vcenter=0, vmax=global_vmax)


from matplotlib.colors import TwoSlopeNorm, SymLogNorm

abs_max = max(abs(global_vmin), abs(global_vmax))

# force symmetric limits
vmin, vmax = -abs_max, abs_max

cmap = plt.get_cmap('seismic', 256)
# choose linthresh very small, e.g. 1% of your full range
linthresh = 0.01 * max(abs(vmin), abs(vmax))

norm = SymLogNorm(
    linthresh=linthresh,
    linscale=1.0,
    vmin=vmin,
    vmax=vmax
)


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 6),
                         sharex=False, sharey=False,
                         gridspec_kw={"right": 0.88,
                                      "hspace": 0.3, "wspace": 0.3})  
# For newer Matplotlib: gridspec_kw={"right":0.88} is equivalent to subplots_adjust(right=0.88).

for idx, key in enumerate(own_samples.keys()):
    ax = axes[idx]
    data = per_method[key]

    sc_t = ax.scatter(
        data["pc1_pcoa_self"], data["pc2_pcoa_self"],
        c=data["logw_self"],  cmap=cmap, norm=norm,
        marker='+', s=50, alpha=0.6, label=key
    )
    ax.scatter(
        data["pc1_pcoa_train"], data["pc2_pcoa_train"],
        c=data["logw_test"], cmap=cmap, norm=norm,
        marker='1', s=50, alpha=0.6, label='real'
    )
    

    pct1_pcoa = data["pct1_pcoa"]
    pct2_pcoa = data["pct2_pcoa"]
    # ax.set_xlabel(f"PC1 ({pct1_pcoa:.1f}% var)", fontsize=18)
    # ax.set_ylabel(f"PC2 ({pct2_pcoa:.1f}% var)", fontsize=18)
    ax.set_title(method_names[key], fontsize=24)
    ax.tick_params(axis='both', labelsize=20)

# Tell fig.colorbar to span all subplots (axes.ravel()) and use a small pad/fraction
cbar = fig.colorbar(
    sc_t,
    ax=axes.ravel().tolist(),
    orientation="vertical",
    fraction=0.02,   # width of colorbar = 2% of the axes box
    pad=0.01         # pad between axes and colorbar = 1% of axes box
)
cbar.set_label("log(w)")

# plt.suptitle("PCoA Scatter for Each Method (real vs. generated sample)", fontsize=18, y=0.98)
plt.tight_layout(rect=[0, 0, 0.88, 0.95])
plt.show()

# %% test credible interval scatterplot
from matplotlib.colors import ListedColormap, BoundaryNorm
BIG = 26   # base size; try 24–30 depending on output medium

plt.rcParams.update({
    "font.size": BIG,
    "axes.titlesize": BIG + 6,
    "axes.labelsize": BIG + 2,
    "xtick.labelsize": BIG,
    "ytick.labelsize": BIG,
    "legend.fontsize": BIG,
    "legend.title_fontsize": BIG + 2,
})
# ---- pick your CI containers ----
logw_lo = log_w_bayesian_CI_025
logw_hi = log_w_bayesian_CI_975


# --- containers ---
per_method = {}

for key in own_samples.keys():
    X_test = dre.to_numpy(sample_test)       # (n_test, k)
    X_self = dre.to_numpy(own_samples[key])  # (n_self, k)

    # CI bounds
    lo_test = logw_lo[key]["test"]
    hi_test = logw_hi[key]["test"]
    lo_self = logw_lo[key]["self"]
    hi_self = logw_hi[key]["self"]

    # categories (0/1/2)
    cat_test = plot_ci_helpers.ci_sign_category(lo_test, hi_test)
    cat_self = plot_ci_helpers.ci_sign_category(lo_self, hi_self)

    # PCoA on stacked compositional data
    stacked_X = np.vstack([X_test, X_self])
    pc1, pc2, pct1, pct2 = help.compute_pcoa_coords(stacked_X)

    n_test = X_test.shape[0]
    per_method[key] = {
        "pc1_pcoa_train": pc1[:n_test],
        "pc2_pcoa_train": pc2[:n_test],
        "pc1_pcoa_self":  pc1[n_test:],
        "pc2_pcoa_self":  pc2[n_test:],
        "cat_train": cat_test,
        "cat_self":  cat_self,
        "pct1_pcoa": pct1,
        "pct2_pcoa": pct2,
    }

# ---- 3-color map (discrete) ----
# choose any 3 distinct colors you like
# cmap3 = ListedColormap(["#2c7bb6", "#bdbdbd", "#d7191c"])  # below / cover / above
cmap3 = ListedColormap(["#2c7bb6", "#010101", "#d7191c"]) 
bounds = [-0.5, 0.5, 1.5, 2.5]
norm3 = BoundaryNorm(bounds, cmap3.N)

fig, axes = plt.subplots(
    nrows=1, ncols=4, figsize=(20, 6),
    sharex=False, sharey=False,
    gridspec_kw={"right": 0.88, "hspace": 0.3, "wspace": 0.3}
)

for idx, key in enumerate(own_samples.keys()):
    ax = axes[idx]
    data = per_method[key]

    sc_self = ax.scatter(
        data["pc1_pcoa_self"], data["pc2_pcoa_self"],
        c=data["cat_self"], cmap=cmap3, norm=norm3,
        marker='+', s=50, alpha=0.7, label='generated'
    )
    ax.scatter(
        data["pc1_pcoa_train"], data["pc2_pcoa_train"],
        c=data["cat_train"], cmap=cmap3, norm=norm3,
        marker='1', s=50, alpha=0.7, label='real'
    )

    # ax.set_xlabel(f"PC1 ({data['pct1_pcoa']:.1f}% var)", fontsize=18)
    # ax.set_ylabel(f"PC2 ({data['pct2_pcoa']:.1f}% var)", fontsize=18)
    ax.set_title(method_names[key], fontsize=24)
    ax.tick_params(axis='both', labelsize=20)

# colorbar with category labels
cbar = fig.colorbar(
    sc_self,
    ax=axes.ravel().tolist(),
    orientation="vertical",
    fraction=0.02,
    pad=0.01,
    ticks=[0, 1, 2]
)
cbar.ax.set_yticklabels([
    "CI < 0",
    "CI covers 0",
    "CI > 0"
])
# cbar.set_label("Credible-interval sign", fontsize=BIG + 2)

plt.tight_layout(rect=[0, 0, 0.88, 0.95])
plt.show()


# %% PCoA for sample_test and sample_self
text_size = 20
plt.rcParams.update({
    'font.size': text_size,             # default text size
    'axes.titlesize': text_size,        # subplot title size
    'axes.labelsize': text_size,        # x/y label size
    'xtick.labelsize': 20,       # x‐tick label size
    'ytick.labelsize': 20,       # y‐tick label size
    'legend.fontsize': text_size,       # legend text
    'legend.title_fontsize': text_size, # legend title
})

fig, axes = plt.subplots(
    nrows=1, ncols=4,
    figsize=(28, 5),   # ← increase height
    sharex=False, sharey=False,
    constrained_layout=True,
    gridspec_kw={"right": 0.88, "wspace": 0.3}
)

# For newer Matplotlib: gridspec_kw={"right":0.88} is equivalent to subplots_adjust(right=0.88).
group_colors = {
    "train":      "tab:blue",
    "generated":  "tab:orange"
}
for idx, key in enumerate(own_samples.keys()):
    ax = axes[idx]
    data = per_method[key]

    sc_t = ax.scatter(
        data["pc1_pcoa_self"], data["pc2_pcoa_self"],
        marker='+', s=30, alpha=0.6, label=key,
        c=group_colors["generated"]
    )
    ax.scatter(
        data["pc1_pcoa_train"], data["pc2_pcoa_train"],
        marker='1', s=30, alpha=0.6, label='real',
        c=group_colors["train"]
    )
    

    pct1_pcoa = data["pct1_pcoa"]
    pct2_pcoa = data["pct2_pcoa"]
    ax.set_xlabel(f"PC1 ({pct1_pcoa:.1f}% var)", fontsize=18)
    ax.set_ylabel(f"PC2 ({pct2_pcoa:.1f}% var)", fontsize=18)
    ax.set_title(method_names[key], fontsize=24)
    ax.tick_params(axis='both', labelsize=20)
    ax.legend(fontsize=22)



# plt.suptitle("PCoA Scatter for Each Method (real vs. generated sample)", fontsize=18, y=0.98)
plt.tight_layout(rect=[0, 0, 0.88, 0.95])
plt.show()
# %% test: plot for credible interval
reload(plot_ci_helpers)


method_keys = list(own_samples.keys())
fig = plot_ci_helpers.compare_methods_ci(
    method_keys,
    log_w_bayesian,
    log_w_bayesian_CI_025,
    log_w_bayesian_CI_975,
    feature_names=None,  # taxa names here
    by_which_method = None,
    top_k=20,
    pos=False,
    legend_fontsize=24,
    legend_outside=True,
    legend_y=0.9
)
plt.show()

# %%  train: raw data for PCoA plot
log_w_results = log_w_bayesian_train
# log_w_results = log_w_bayesian_CI_025_train
# log_w_results = log_w_bayesian_CI_975_train


method_names = {
    "d":       "Dirichlet",
    "dt":      "Dirichlet Tree",
    "icfm":  "ICFM",
    "mbgan":   "MB-GAN"
}

from sklearn.cross_decomposition import CCA
from scipy.spatial.distance import pdist, squareform


# Containers to hold all per-method data
per_method = {}
all_logw_values = []  # we'll collect every log(w) to determine global vmin/vmax
metric='braycurtis' # for PCoA plot


for key in own_samples.keys():

    # Convert compositional arrays to NumPy
    X_train = dre.to_numpy(sample_train)       # shape: (n_train, k)
    X_test  = dre.to_numpy(sample_test)        # shape: (n_test, k)
    X_self  = dre.to_numpy(own_samples[key])   # shape: (n_self, k)

    logw_train = log_w_results[key]["train"]  # shape: (n_train,)
    logw_self  = log_w_results[key]["self"]   # shape: (n_self,)

    # Collect for global vmin/vmax
    all_logw_values.append(logw_train)
    all_logw_values.append(logw_self)

    # compute CCA
    stacked_X = np.vstack([X_train,X_self])
    stacked_Y = np.vstack((logw_train[:, None],
                       logw_self[:,  None]))


    # PCoA
    pc1, pc2, pct1, pct2 = help.compute_pcoa_coords(stacked_X)
    pc1_pcoa_train = pc1[:n_train]
    pc1_pcoa_self = pc1[n_train:]
    pc2_pcoa_train = pc2[:n_train]
    pc2_pcoa_self = pc2[n_train:]


    # Save everything for this method
    per_method[key] = {
        "logw_train":     logw_train,
        "logw_self":      logw_self,
        "pc1_pcoa_train":   pc1_pcoa_train,
        "pc1_pcoa_self":    pc1_pcoa_self,
        "pc2_pcoa_train":   pc2_pcoa_train,
        "pc2_pcoa_self":    pc2_pcoa_self,
        "pct1_pcoa":    pct1,
        "pct2_pcoa":    pct2,
    }

# Stack all log(w) values to find a global range
all_logw_flat = np.concatenate(all_logw_values)
# global_vmin = np.min(all_logw_flat)
# global_vmax = np.max(all_logw_flat)
# for training
global_vmin = -30
global_vmax = 30

# for testing
# global_vmin = -12
# global_vmax = 12
cmap = plt.get_cmap('seismic')
norm = TwoSlopeNorm(vmin=global_vmin, vcenter=0, vmax=global_vmax)


# PCoA scatter plots

BIG = 26   # base size; try 24–30 depending on output medium

plt.rcParams.update({
    "font.size": BIG,
    "axes.titlesize": BIG + 6,
    "axes.labelsize": BIG + 2,
    "xtick.labelsize": BIG,
    "ytick.labelsize": BIG,
    "legend.fontsize": BIG,
    "legend.title_fontsize": BIG + 2,
})

# import matplotlib.colors
# import importlib
# importlib.reload(matplotlib.colors)

from matplotlib.colors import TwoSlopeNorm, SymLogNorm

abs_max = max(abs(global_vmin), abs(global_vmax))

# force symmetric limits
vmin, vmax = -abs_max, abs_max

cmap = plt.get_cmap('seismic', 256)
# choose linthresh very small, e.g. 1% of your full range
linthresh = 0.01 * max(abs(vmin), abs(vmax))

norm = SymLogNorm(
    linthresh=linthresh,
    linscale=1.0,
    vmin=vmin,
    vmax=vmax
)


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 6),
                         sharex=False, sharey=False,
                         gridspec_kw={"right": 0.88,
                                      "hspace": 0.3, "wspace": 0.3})  
# For newer Matplotlib: gridspec_kw={"right":0.88} is equivalent to subplots_adjust(right=0.88).

for idx, key in enumerate(own_samples.keys()):
    ax = axes[idx]
    data = per_method[key]

    sc_t = ax.scatter(
        data["pc1_pcoa_train"], data["pc2_pcoa_train"],
        c=data["logw_train"], cmap=cmap, norm=norm,
        marker='1', s=30, alpha=0.8, label='real'
    )
    ax.scatter(
        data["pc1_pcoa_self"], data["pc2_pcoa_self"],
        c=data["logw_self"],  cmap=cmap, norm=norm,
        marker='+', s=30, alpha=0.8, label=key
    )

    pct1_pcoa = data["pct1_pcoa"]
    pct2_pcoa = data["pct2_pcoa"]

    # ax.set_xlabel(f"PC1 ({pct1_pcoa:.1f}% var)", fontsize=18)
    # ax.set_ylabel(f"PC2 ({pct2_pcoa:.1f}% var)", fontsize=18)
    ax.set_title(method_names[key], fontsize=24)
    ax.tick_params(axis='both', labelsize=20)

# Tell fig.colorbar to span all subplots (axes.ravel()) and use a small pad/fraction
cbar = fig.colorbar(
    sc_t,
    ax=axes.ravel().tolist(),
    orientation="vertical",
    fraction=0.02,   # width of colorbar = 2% of the axes box
    pad=0.01         # pad between axes and colorbar = 1% of axes box
)
cbar.set_label("log(w)")

# plt.suptitle("PCoA Scatter for Each Method (real vs. generated sample)", fontsize=18, y=0.98)
plt.tight_layout(rect=[0, 0, 0.88, 0.95])
plt.show()
# %% train credible interval scatterplot
from matplotlib.colors import ListedColormap, BoundaryNorm
BIG = 26   # base size; try 24–30 depending on output medium

plt.rcParams.update({
    "font.size": BIG,
    "axes.titlesize": BIG + 6,
    "axes.labelsize": BIG + 2,
    "xtick.labelsize": BIG,
    "ytick.labelsize": BIG,
    "legend.fontsize": BIG,
    "legend.title_fontsize": BIG + 2,
})
# ---- pick your CI containers ----
logw_lo = log_w_bayesian_CI_025_train
logw_hi = log_w_bayesian_CI_975_train


# --- containers ---
per_method = {}

for key in own_samples.keys():
    X_train = dre.to_numpy(sample_train)       # (n_train, k)
    X_self = dre.to_numpy(own_samples[key])  # (n_self, k)

    # CI bounds
    lo_train = logw_lo[key]["train"]
    hi_train = logw_hi[key]["train"]
    lo_self = logw_lo[key]["self"]
    hi_self = logw_hi[key]["self"]

    # categories (0/1/2)
    cat_train = plot_ci_helpers.ci_sign_category(lo_train, hi_train)
    cat_self = plot_ci_helpers.ci_sign_category(lo_self, hi_self)

    # PCoA on stacked compositional data
    stacked_X = np.vstack([X_train, X_self])
    pc1, pc2, pct1, pct2 = help.compute_pcoa_coords(stacked_X)

    n_train = X_train.shape[0]
    per_method[key] = {
        "pc1_pcoa_train": pc1[:n_train],
        "pc2_pcoa_train": pc2[:n_train],
        "pc1_pcoa_self":  pc1[n_train:],
        "pc2_pcoa_self":  pc2[n_train:],
        "cat_train": cat_train,
        "cat_self":  cat_self,
        "pct1_pcoa": pct1,
        "pct2_pcoa": pct2,
    }

# ---- 3-color map (discrete) ----
# choose any 3 distinct colors you like
# cmap3 = ListedColormap(["#2c7bb6", "#bdbdbd", "#d7191c"])  # below / cover / above
# cmap3 = ListedColormap(["#6C8EA5", "#d7191c", "#F2C14E",])  # below / cover / above
# cmap3 = ListedColormap(["#4C8DAE", "#F2C14E", "#B36A5E"])
# cmap3 = ListedColormap(["#6C8EA5", "#F4D35E", "#9B6A8B"])
cmap3 = ListedColormap(["#2c7bb6", "#010101", "#d7191c"]) 

bounds = [-0.5, 0.5, 1.5, 2.5]
norm3 = BoundaryNorm(bounds, cmap3.N)

fig, axes = plt.subplots(
    nrows=1, ncols=4, figsize=(20, 6),
    sharex=False, sharey=False,
    gridspec_kw={"right": 0.88, "hspace": 0.3, "wspace": 0.3}
)

for idx, key in enumerate(own_samples.keys()):
    ax = axes[idx]
    data = per_method[key]

    sc_self = ax.scatter(
        data["pc1_pcoa_self"], data["pc2_pcoa_self"],
        c=data["cat_self"], cmap=cmap3, norm=norm3,
        marker='+', s=50, alpha=0.7, label='generated'
    )
    ax.scatter(
        data["pc1_pcoa_train"], data["pc2_pcoa_train"],
        c=data["cat_train"], cmap=cmap3, norm=norm3,
        marker='1', s=50, alpha=0.7, label='real'
    )

    # ax.set_xlabel(f"PC1 ({pct1_pcoa:.1f}% var)", fontsize=18)
    # ax.set_ylabel(f"PC2 ({pct2_pcoa:.1f}% var)", fontsize=18)
    ax.set_title(method_names[key], fontsize=24)
    ax.tick_params(axis='both', labelsize=20)

# colorbar with category labels
cbar = fig.colorbar(
    sc_self,
    ax=axes.ravel().tolist(),
    orientation="vertical",
    fraction=0.02,
    pad=0.01,
    ticks=[0, 1, 2]
)
cbar.ax.set_yticklabels([
    "CI < 0",
    "CI covers 0",
    "CI > 0"
])
# cbar.set_label("Credible-interval sign", fontsize=BIG + 2)

plt.tight_layout(rect=[0, 0, 0.88, 0.95])
plt.show()


# %% train: plot for credible interval
method_keys = list(own_samples.keys())
fig = plot_ci_helpers.compare_methods_ci(
    method_keys,
    log_w_bayesian_train,
    log_w_bayesian_CI_025_train,
    log_w_bayesian_CI_975_train,
    feature_names=None,  # taxa names here
    by_which_method = None,
    top_k=20,
    pos=False,
    legend_fontsize=24,
    legend_outside=True,
    legend_y=0.9
)
plt.show()
# %%
