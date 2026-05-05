""" "
Miscellaneous functions to plot.

Date: September 2018
Last updated: March 2026
Original Author: Ignacio Heredia (CSIC)
Updated and maintained by: Wout Decrop (VLIZ)
Contact: wout.decrop@vliz.be
Github: ai4os-hub / phyto-plankton-classification
"""

import json
import os

import matplotlib.pylab as plt
import numpy as np
import seaborn

from planktonclass import paths


def create_pred_path(save_path, dir="", weighted=False, **kwargs):
    """
    Create the directory path for saving the plots based on the provided options.

    Args:
        save_path (str): Path where the plots will be saved.
        paths (object): Object with timestamped directory creation method.
        aimed (bool): Flag indicating whether the confusion matrices are aimed or not.
        weighted (bool): Flag indicating whether to compute weighted confusion matrices.

    Returns:
        str: Directory path for saving the plots.
    """
    value = next(iter(kwargs.values()))
    if weighted:
        pred_path = save_path or os.path.join(paths.get_results_dir(), dir,
                                              "confusion_weighted")
    else:
        pred_path = save_path or os.path.join(paths.get_results_dir(), dir,
                                              value)

    os.makedirs(pred_path, exist_ok=True)
    return pred_path


def plt_conf_matrix(conf_mat, labels=False, normalized=False):
    num_classes = conf_mat.shape[0]
    fig_size = min(max(12, num_classes * 0.28), 34)
    tick_fontsize = min(max(5, 12 - (num_classes // 12)), 9)
    cbar_kws = {"fraction": 0.035, "pad": 0.03}

    fig, ax = plt.subplots(figsize=(fig_size, fig_size), facecolor="white")
    hm = seaborn.heatmap(
        conf_mat,
        annot=False,
        square=True,
        cmap="Blues",
        linewidths=0.35,
        linecolor="#d9d9d9",
        vmin=0,
        vmax=1 if normalized else None,
        cbar_kws=cbar_kws,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    fontsize = tick_fontsize
    hm.yaxis.set_ticklabels(
        hm.yaxis.get_ticklabels(),
        rotation=0,
        ha="right",
        fontsize=fontsize,
    )
    hm.xaxis.set_ticklabels(
        hm.xaxis.get_ticklabels(),
        rotation=90,
        ha="right",
        fontsize=fontsize,
    )

    ax.set_facecolor("white")
    ax.tick_params(axis="both", length=0)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_xlabel("Predicted Class", fontsize=12)

    colorbar = hm.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=max(fontsize - 1, 5), length=0)

    return fig, ax


def training_plots(conf, stats, show_val=True, show_ckpt=True):
    """
    Plot the loss and accuracy metrics for a timestamped training.

    Parameters
    ----------
    conf : dict
        Configuration dict
    stats : dict
        Statistics dict
    show_val: bool
        Plot the validation data if available
    show_ckpt : bool
        Plot epochs at which ckpts have been made
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Training
    axs[0].plot(stats["epoch"], stats["loss"], label="Training")
    axs[1].plot(stats["epoch"], stats["accuracy"], label="Training")

    # Validation
    if (conf["training"]["use_validation"]) and show_val:
        axs[0].plot(stats["epoch"], stats["val_loss"], label="Validation")
        axs[1].plot(stats["epoch"], stats["val_accuracy"], label="Validation")

    # Model Checkpoints
    if (conf["training"]["ckpt_freq"] is not None) and show_ckpt:
        period = max(
            1,
            int(conf["training"]["ckpt_freq"] * conf["training"]["epochs"]),
        )
        ckpts = np.arange(0, conf["training"]["epochs"], period)
        for i, c in enumerate(ckpts):
            label = None
            if i == 0:
                label = "checkpoints"
            axs[0].axvline(c, linestyle="--", color="#f9d1e0")
            axs[1].axvline(c, linestyle="--", color="#f9d1e0", label=label)

    axs[1].set_ylim([0, 1])
    axs[0].set_xlabel("Epochs"), axs[0].set_title("Loss")
    axs[1].set_xlabel("Epochs"), axs[1].set_title("Accuracy")
    axs[0].legend(loc="upper right")


def multi_training_plots(timestamps, legend_loc="upper right"):
    """
    Compare the loss and accuracy metrics for a timestamped training.

    Parameters
    ----------
    timestamps : str, or list of strs
        Configuration dict
    legend_loc: str
        Legend position
    """
    if timestamps is str:
        timestamps = [timestamps]

    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    axs = axs.flatten()

    for ts in timestamps:

        # Set the timestamp
        paths.timestamp = ts

        # Load training statistics
        stats_path = os.path.join(paths.get_stats_dir(), "stats.json")
        with open(stats_path) as f:
            stats = json.load(f)

        # Load training configuration
        conf_path = os.path.join(paths.get_conf_dir(), "conf.json")
        with open(conf_path) as f:
            conf = json.load(f)

        # Training
        axs[0].plot(stats["epoch"], stats["loss"], label=ts)
        axs[1].plot(stats["epoch"], stats["accuracy"], label=ts)

        # Validation
        if conf["training"]["use_validation"]:
            axs[2].plot(stats["epoch"], stats["val_loss"], label=ts)
            axs[3].plot(stats["epoch"], stats["val_accuracy"], label=ts)

    axs[1].set_ylim([0, 1])
    axs[3].set_ylim([0, 1])

    for i in range(4):
        axs[0].set_xlabel("Epochs")

    axs[0].set_title("Training Loss")
    axs[1].set_title("Training Accuracy")
    axs[2].set_title("Validation Loss")
    axs[3].set_title("Validation Accuracy")

    axs[0].legend(loc=legend_loc)
