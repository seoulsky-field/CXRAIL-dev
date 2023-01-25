import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from typing import Optional
import os


class TestMetricsReporter:
    def __init__(self, hydra_cfg, preds, targets, target_columns=None):
        self.preds = preds
        self.targets = targets
        self.hydra_cfg = hydra_cfg
        self.target_columns = target_columns
        self.save_dir = os.path.join(hydra_cfg.log_dir, "images")


class AUROCMetricReporter(TestMetricsReporter):
    def __init__(self, hydra_cfg, preds, targets, target_columns=None):
        super().__init__(hydra_cfg, preds, targets, target_columns)

    def get_class_auroc_score(self):
        return roc_auc_score(self.targets, self.preds, average=None)

    def get_macro_auroc_score(self, targets: Optional = None, preds: Optional = None):
        targets = self.targets if targets is None else targets
        preds = self.preds if preds is None else preds

        return roc_auc_score(targets, preds, average="macro")

    def get_micro_auroc_score(self):
        return roc_auc_score(self.targets, self.preds, average="micro")

    def get_auroc_details(self, targets, preds):
        fpr, tpr, thresholds = roc_curve(targets, preds)
        J = tpr - fpr
        ix = np.argmax(J)
        best_threshold = thresholds[ix]

        return fpr, tpr, thresholds, ix

    def plot_class_auroc_details(
        self, targets, preds, col_name, overlap=False, color="black"
    ):
        fpr, tpr, thresholds, ix = self.get_auroc_details(targets, preds)
        best_threshold = thresholds[ix]
        sensitivity, specificity = tpr[ix], 1 - fpr[ix]

        macro_auroc_score = self.get_macro_auroc_score(targets, preds)

        if not overlap:
            plt.clf()

        plt.title(f"{col_name} ROC Curve", fontweight="bold")

        plt.plot([0, 1], [0, 1], linestyle="--", markersize=0.05, color="black")
        plt.plot(
            fpr,
            tpr,
            marker=".",
            color=color,
            markersize=0.05,
            label=f"{col_name} AUROC: {macro_auroc_score:>.4f}",
        )

        details = f"Best Threshold: {best_threshold:>.4f}\nSensitivity: {sensitivity:>.4f}\nSpecificity: {specificity:>.4f}"
        plt.scatter(
            fpr[ix],
            tpr[ix],
            marker="+",
            s=100,
            color="r",
            label=details if not overlap else None,
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")

        if not overlap:
            plt.savefig(
                os.path.join(self.save_dir, f"{col_name}_roc_curve.png"), dpi=300
            )

    def plot_overlap_roc_curve(self):
        # target_columns = *self.hydra_cfg.Dataset.train_cols
        # target_columns = list(self.hydra_cfg.Dataset.train_cols)
        colors = ["orange", "green", "blue", "purple", "pink"]

        plt.clf()
        for idx in range(5):
            self.plot_class_auroc_details(
                self.targets[:, idx],
                self.preds[:, idx],
                col_name=self.target_columns[idx],
                overlap=True,
                color=colors[idx],
            )

        plt.title("All Classes' ROC CURVE", fontweight="bold")
        plt.savefig(os.path.join(self.save_dir, "overlap_roc_curve.png"), dpi=300)
