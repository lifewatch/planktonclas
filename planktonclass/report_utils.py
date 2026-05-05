"""
Utilities to generate evaluation reports and plots for trained runs.
"""

import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from planktonclass import paths, plot_utils


REPORT_THRESHOLD_LEVELS = np.array([0.0, 0.50, 0.75, 0.90, 0.95], dtype=float)


def _latest_timestamp(models_dir):
    timestamps = sorted(
        [
            name
            for name in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, name))
        ]
    )
    if not timestamps:
        raise FileNotFoundError(f"No trained models found in {models_dir}")
    return timestamps[-1]


def _display_path(path):
    try:
        return os.path.relpath(path, paths.get_base_dir()).replace("\\", "/")
    except ValueError:
        return path


def _find_predictions_file(predictions_dir):
    if not os.path.isdir(predictions_dir):
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")

    candidates = sorted(
        [
            name
            for name in os.listdir(predictions_dir)
            if name.endswith(".json") and "DS_split" in name
        ]
    )
    if not candidates:
        raise FileNotFoundError(
            f"No saved test predictions found in {predictions_dir}. "
            "Run training with testing enabled first."
        )
    return os.path.join(predictions_dir, candidates[-1])


def _save_training_plot(stats, conf, results_dir):
    plot_utils.training_plots(conf=conf, stats=stats)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_metrics.png"), dpi=200)
    plt.close()


def _save_confusion_plots(true_labels, pred_labels, class_names, results_dir):
    cm = confusion_matrix(true_labels, pred_labels)
    fig, _ = plot_utils.plt_conf_matrix(cm, labels=class_names, normalized=False)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "confusion_matrix_counts.png"), dpi=200)
    plt.close(fig)

    cm_norm = confusion_matrix(true_labels, pred_labels, normalize="true")
    fig, _ = plot_utils.plt_conf_matrix(
        np.round(cm_norm, 3), labels=class_names, normalized=True
    )
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "confusion_matrix_normalized.png"), dpi=200)
    plt.close(fig)


def _save_topk_plot(true_labels, pred_lab, results_dir):
    ks = [1, 2, 3, 4, 5]
    accuracies = []
    pred_array = np.array(pred_lab)
    for k in ks:
        topk = min(k, pred_array.shape[1])
        mask = [lab in pred_array[i, :topk] for i, lab in enumerate(true_labels)]
        accuracies.append(float(np.mean(mask)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, np.array(accuracies) * 100, marker="o", linestyle="-", color="#007acc", linewidth=2)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Top-K")
    ax.set_title("Top-K Accuracy (K=1 to 5)")
    for idx, value in enumerate(accuracies):
        ax.text(ks[idx], value * 100 + 0.5, f"{value * 100:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "topk_accuracy.png"), dpi=200)
    plt.close(fig)

    marginal = np.diff(np.array(accuracies) * 100, prepend=(np.array(accuracies) * 100)[0])
    marginal[0] = np.nan
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(ks, marginal, color="#ffa726", edgecolor="black")
    for bar, inc in zip(bars, marginal):
        if not np.isnan(inc):
            ax.text(bar.get_x() + bar.get_width() / 2, inc + 0.2, f"{inc:.2f}%", ha="center", fontsize=10)
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Marginal Increase in Accuracy (%)")
    ax.set_title("Marginal Accuracy Gain per K")
    ax.set_xticks(ks)
    ax.set_ylim(0, max(marginal[1:]) + 2 if len(marginal) > 1 else 1)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "topk_marginal_increase.png"), dpi=200)
    plt.close(fig)

    return {
        "top1_accuracy": accuracies[0],
        "top3_accuracy": accuracies[2],
        "top5_accuracy": accuracies[4],
    }


def _save_class_report(true_labels, pred_labels, class_names, results_dir):
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=[str(name) for name in class_names],
        output_dict=True,
        zero_division=0,
    )

    report_csv = os.path.join(results_dir, "classification_report.csv")
    with open(report_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "precision", "recall", "f1-score", "support"])
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                writer.writerow(
                    [
                        label,
                        metrics.get("precision", ""),
                        metrics.get("recall", ""),
                        metrics.get("f1-score", ""),
                        metrics.get("support", ""),
                    ]
                )

    return report


def _save_per_class_plot(true_labels, pred_labels, class_names, results_dir):
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )

    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 1.2), 6))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "per_class_metrics.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 1.2), 5))
    ax.bar(x, support, color="#6c757d")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Samples")
    ax.set_title("Test Support Per Class")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "class_support.png"), dpi=200)
    plt.close(fig)


def _build_prediction_dataframe(predictions, class_names):
    pred_lab = np.array(predictions["pred_lab"], dtype=int)
    pred_prob = np.array(predictions["pred_prob"], dtype=float)
    true_lab = np.array(predictions["true_lab"], dtype=int)

    if "pred_lab_names" in predictions:
        top1_names = np.array([row[0] for row in predictions["pred_lab_names"]], dtype=str)
    else:
        top1_names = np.array([class_names[idx] for idx in pred_lab[:, 0]], dtype=str)

    true_names = np.array([class_names[idx] for idx in true_lab], dtype=str)

    data = pd.DataFrame(
        {
            "true_label": true_names,
            "top1": top1_names,
            "probability": pred_prob[:, 0],
        }
    )
    data["top1_correct"] = data["true_label"] == data["top1"]
    return data


def _save_threshold_confusion_plots(data, results_dir):
    class_names = np.unique(data["top1"])
    threshold_dir = os.path.join(results_dir, "confusion_thresholds")
    threshold_dir_weighted = os.path.join(results_dir, "confusion_thresholds_weighted")
    os.makedirs(threshold_dir, exist_ok=True)
    os.makedirs(threshold_dir_weighted, exist_ok=True)

    for weighted, save_dir in [(False, threshold_dir), (True, threshold_dir_weighted)]:
        for threshold in REPORT_THRESHOLD_LEVELS:
            subset = data[data["probability"] > threshold]
            if subset.empty:
                continue
            weights = np.array(subset["probability"]) if weighted else None
            conf_mat = confusion_matrix(
                np.array(subset["true_label"]).astype(str),
                np.array(subset["top1"]).astype(str),
                labels=class_names,
                sample_weight=weights,
            )
            row_sums = conf_mat.sum(axis=1, keepdims=True)
            normed = np.divide(conf_mat, row_sums, out=np.zeros_like(conf_mat, dtype=float), where=row_sums != 0)
            fig, _ = plot_utils.plt_conf_matrix(
                normed, labels=class_names, normalized=True
            )
            fig.tight_layout()
            suffix = f"weighted_{threshold:.2f}" if weighted else f"{threshold:.2f}"
            fig.savefig(os.path.join(save_dir, f"confusion_matrix_{suffix}.png"), dpi=200)
            plt.close(fig)


def _compute_metrics_for_subset(slice_data, class_name):
    tp = np.sum((slice_data["true_label"] == class_name) & (slice_data["top1"] == class_name))
    fp = np.sum((slice_data["true_label"] != class_name) & (slice_data["top1"] == class_name))
    fn = np.sum((slice_data["true_label"] == class_name) & (slice_data["top1"] != class_name))
    tn = np.sum((slice_data["true_label"] != class_name) & (slice_data["top1"] != class_name))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    outer_precision = tp / np.sum(slice_data["top1"] == class_name) if np.sum(slice_data["top1"] == class_name) else 0.0
    outer_recall = tp / np.sum(slice_data["true_label"] == class_name) if np.sum(slice_data["true_label"] == class_name) else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    y_true_binary = (slice_data["true_label"] == class_name).astype(int)
    if len(np.unique(y_true_binary)) > 1:
        auc = roc_auc_score(y_true_binary, slice_data["probability"])
    else:
        auc = 0.0

    return precision, recall, outer_precision, outer_recall, f1, accuracy, specificity, fpr, fnr, auc


def _save_cutoff_evolution_plots(data, results_dir):
    cutoff_dir = os.path.join(results_dir, "cut_off_changes")
    os.makedirs(cutoff_dir, exist_ok=True)
    thresholds = REPORT_THRESHOLD_LEVELS

    for class_name in np.unique(data["top1"]):
        metrics = []
        data_slice_ratio = []
        total_for_class = max(np.sum(data["top1"] == class_name), 1)

        for threshold in thresholds:
            subset = data[data["probability"] > threshold]
            data_slice_ratio.append(np.sum(subset["top1"] == class_name) / total_for_class if total_for_class else 0.0)
            metrics.append(_compute_metrics_for_subset(subset, class_name) if not subset.empty else (0.0,) * 10)

        fig, ax = plt.subplots(figsize=(12, 8))
        cmap = plt.get_cmap("tab10")
        colors = cmap(np.linspace(0, 1, 12))
        ax.plot(thresholds, [m[0] for m in metrics], label="Precision", linestyle="-", marker="o", color=colors[0])
        ax.plot(thresholds, [m[1] for m in metrics], label="Recall", linestyle="--", marker="s", color=colors[1])
        ax.plot(thresholds, data_slice_ratio, label="Loss of Predictions", linestyle="-", marker="*", color=colors[4])
        ax.plot(thresholds, [m[2] for m in metrics], label="Outer Precision", linestyle=":", marker="^", color=colors[2])
        ax.plot(thresholds, [m[3] for m in metrics], label="Outer Recall", linestyle="-.", marker="D", color=colors[3])
        ax.plot(thresholds, [m[4] for m in metrics], label="F1 Score", linestyle="-", marker="x", color=colors[5])
        ax.plot(thresholds, [m[5] for m in metrics], label="Accuracy", linestyle="--", marker="o", color=colors[6])
        ax.plot(thresholds, [m[6] for m in metrics], label="Specificity", linestyle=":", marker="s", color=colors[7])
        ax.plot(thresholds, [m[7] for m in metrics], label="False Positive Rate", linestyle="-.", marker="^", color=colors[8])
        ax.plot(thresholds, [m[8] for m in metrics], label="False Negative Rate", linestyle="-", marker="D", color=colors[9])
        ax.plot(thresholds, [m[9] for m in metrics], label="AUC-ROC", linestyle="--", marker="*", color=colors[10])
        ax.legend(loc="best", fontsize="medium")
        ax.set_xlabel("Cutoff value for probability")
        ax.set_ylabel("Metric Score")
        ax.set_title(class_name)
        ax.grid(True, linestyle="--", alpha=0.7)
        fig.tight_layout()
        safe_name = class_name.replace("/", "+")
        fig.savefig(os.path.join(cutoff_dir, f"{safe_name}.png"), dpi=200)
        plt.close(fig)


def _save_classwise_progression_plots(data, results_dir):
    progression_dir = os.path.join(results_dir, "proportional_progression")
    os.makedirs(progression_dir, exist_ok=True)
    thresholds = REPORT_THRESHOLD_LEVELS

    for class_name in np.unique(data["top1"]):
        base_counts = pd.DataFrame(data[data["top1"] == class_name]["true_label"].value_counts())
        if base_counts.empty:
            continue
        for idx, threshold in enumerate(thresholds):
            subset = data[data["probability"] > threshold]
            counts = subset[subset["top1"] == class_name]["true_label"].value_counts()
            base_counts[str(idx)] = counts

        if "count" in base_counts.columns:
            base_counts = base_counts.drop("count", axis=1)

        proportional = base_counts.div(base_counts.sum(axis=0), axis=1)
        proportional = proportional[proportional > 0.01] * 100
        absolute = base_counts[proportional > 0.01]
        proportional.columns = thresholds
        absolute.columns = thresholds

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
        handles, labels = proportional.dropna(how="all").transpose().plot.area(
            ax=axes[0],
            xlabel="Cutoff value for probability",
            legend=False,
            title="Proportional progression of TRUE class distribution",
        ).get_legend_handles_labels()
        axes[0].set_ylabel("Class proportion (%)")
        absolute.dropna(how="all").transpose().plot.area(
            ax=axes[1],
            xlabel="Cutoff value for probability",
            legend=False,
            title="Absolute progression of TRUE class distribution",
        )
        axes[1].set_ylabel("Class count")
        fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=10, bbox_to_anchor=(0.5, 0.02))
        fig.tight_layout(rect=[0, 0.15, 1, 0.95])
        safe_name = class_name.replace("/", "+")
        fig.savefig(os.path.join(progression_dir, f"{safe_name}.png"), dpi=200)
        plt.close(fig)


def _define_cutoff(data, class_name, precision_target=0.95):
    subset = data[data["top1"] == class_name].sort_values(by="probability", ascending=False).reset_index(drop=True)
    if subset.empty:
        return np.inf
    precision_curve = np.cumsum(subset["top1_correct"]) / (subset.index + 1)
    acceptable = np.where(precision_curve > precision_target)[0]
    if len(acceptable) > 0:
        return float(np.min(subset["probability"].iloc[acceptable]))
    return np.inf


def _apply_mask(data, mask_data):
    mask = np.repeat(False, len(data))
    for _, row in mask_data.iterrows():
        mask |= ((data["top1"] == row["Class"]) & (data["probability"] >= row["probability_cutoff"])).to_numpy()
    return data[mask]


def _masked_data_for_target(data, precision_target):
    mask = [[class_name, _define_cutoff(data, class_name, precision_target=precision_target)] for class_name in np.unique(data["top1"])]
    mask_data = pd.DataFrame(mask, columns=["Class", "probability_cutoff"])
    return _apply_mask(data, mask_data)


def _save_threshold_summary_plot(data, results_dir):
    summary_dir = os.path.join(results_dir, "summary_threshold")
    os.makedirs(summary_dir, exist_ok=True)
    thresholds = REPORT_THRESHOLD_LEVELS[1:]

    acc = np.zeros(len(thresholds))
    pr_weighted = np.zeros(len(thresholds))
    rec_weighted = np.zeros(len(thresholds))
    f1_weighted = np.zeros(len(thresholds))
    dataset_prop = np.zeros(len(thresholds))
    class_prop = np.zeros(len(thresholds))

    all_classes = len(np.unique(data["top1"]))
    for i, precision_target in enumerate(thresholds):
        masked_data = _masked_data_for_target(data, precision_target)
        if masked_data.empty:
            continue
        dataset_prop[i] = len(masked_data) / len(data)
        class_prop[i] = len(np.unique(masked_data["top1"])) / all_classes if all_classes else 0.0
        acc[i] = np.mean(masked_data["true_label"] == masked_data["top1"])
        pr_weighted[i] = precision_recall_fscore_support(masked_data["true_label"], masked_data["top1"], average="weighted", zero_division=0)[0]
        rec_weighted[i] = precision_recall_fscore_support(masked_data["true_label"], masked_data["top1"], average="weighted", zero_division=0)[1]
        f1_weighted[i] = precision_recall_fscore_support(masked_data["true_label"], masked_data["top1"], average="weighted", zero_division=0)[2]

    show_data = pd.DataFrame(
        {
            "Accuracy": acc,
            "Precision (weighted)": pr_weighted,
            "Recall (weighted)": rec_weighted,
            "F1 Score (weighted)": f1_weighted,
            "Dataset Proportion": dataset_prop,
            "Class Proportion": class_prop,
        },
        index=thresholds,
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, metric in enumerate(show_data.columns):
        if metric in ["Dataset Proportion", "Class Proportion"]:
            ax.plot(show_data.index, show_data[metric], marker="o", linestyle="--", color=colors[i], label=metric, alpha=0.7)
        elif metric == "Accuracy":
            ax.plot(show_data.index, show_data[metric], marker="x", color=colors[i], label=metric, alpha=1, markersize=10)
        else:
            ax.plot(show_data.index, show_data[metric], marker="o", color=colors[i], label=metric, alpha=0.7)
    ax.set_title("Scores after masking increasing thresholds")
    ax.set_xlabel("Probability Threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(summary_dir, "summary_threshold_plot.png"), dpi=200)
    plt.close(fig)


def generate_report(timestamp=None, progress=None, mode="quick"):
    progress = progress or (lambda message: None)
    models_dir = paths.get_models_dir()
    if timestamp is None:
        timestamp = _latest_timestamp(models_dir)

    paths.timestamp = timestamp
    results_dir = paths.get_results_dir()
    os.makedirs(results_dir, exist_ok=True)
    progress(f"Preparing report for: {timestamp}")
    progress(f"Results directory: {_display_path(results_dir)}")

    stats_path = os.path.join(paths.get_stats_dir(), "stats.json")
    conf_path = os.path.join(paths.get_conf_dir(), "conf.json")
    class_path = os.path.join(paths.get_ts_splits_dir(), "classes.txt")
    predictions_path = _find_predictions_file(paths.get_predictions_dir())
    progress(f"Loading stats: {_display_path(stats_path)}")
    progress(f"Loading config: {_display_path(conf_path)}")
    progress(f"Loading predictions: {_display_path(predictions_path)}")

    with open(stats_path) as f:
        stats = json.load(f)
    with open(conf_path) as f:
        conf = json.load(f)
    with open(predictions_path) as f:
        predictions = json.load(f)

    class_names = np.genfromtxt(class_path, dtype="str", delimiter="/n")
    true_labels = np.array(predictions["true_lab"], dtype=int)
    pred_lab = np.array(predictions["pred_lab"], dtype=int)
    pred_labels = pred_lab[:, 0]
    data = _build_prediction_dataframe(predictions=predictions, class_names=class_names)

    progress("Creating training metric plots")
    _save_training_plot(stats=stats, conf=conf, results_dir=results_dir)
    progress("Creating confusion matrices")
    _save_confusion_plots(
        true_labels=true_labels,
        pred_labels=pred_labels,
        class_names=class_names,
        results_dir=results_dir,
    )
    progress("Creating top-k accuracy plots")
    topk_summary = _save_topk_plot(
        true_labels=true_labels,
        pred_lab=pred_lab,
        results_dir=results_dir,
    )
    progress("Writing classification report")
    class_report = _save_class_report(
        true_labels=true_labels,
        pred_labels=pred_labels,
        class_names=class_names,
        results_dir=results_dir,
    )
    progress("Creating per-class metric plots")
    _save_per_class_plot(
        true_labels=true_labels,
        pred_labels=pred_labels,
        class_names=class_names,
        results_dir=results_dir,
    )
    if mode == "full":
        progress("Creating threshold confusion plots")
        _save_threshold_confusion_plots(data=data, results_dir=results_dir)
        progress("Creating cutoff evolution plots")
        _save_cutoff_evolution_plots(data=data, results_dir=results_dir)
        progress("Creating classwise progression plots")
        _save_classwise_progression_plots(data=data, results_dir=results_dir)
        progress("Creating threshold summary plot")
        _save_threshold_summary_plot(data=data, results_dir=results_dir)

    summary = {
        "timestamp": timestamp,
        "mode": mode,
        "results_dir": results_dir,
        "predictions_file": predictions_path,
        **topk_summary,
        "macro_f1": class_report["macro avg"]["f1-score"],
        "weighted_f1": class_report["weighted avg"]["f1-score"],
    }

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    progress(f"Summary saved to: {_display_path(os.path.join(results_dir, 'summary.json'))}")

    return summary
