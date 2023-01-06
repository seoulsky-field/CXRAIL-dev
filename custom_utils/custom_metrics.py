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


def accuracy(pred, y):
    correct = 0
    num_samples = y.shape[0]
    pred = np.where(pred < 0.5, 0.0, pred)
    pred = np.where(pred >= 0.5, 1.0, pred)
    correct += (pred == y).sum().item()
    acc = 100 * correct / num_samples

    return acc


def get_accuracy_list(preds, labels):
    accuracy_list = []
    num_labels = labels.shape[1]
    for i in range(num_labels):
        acc = accuracy(preds[:, i], labels[:, i])
        accuracy_list.append(acc)

    return accuracy_list


def roc_curves(preds, labels, train_cols):
    fpr, tpr = dict(), dict()
    roc_auc_scores = dict()

    for name in train_cols:
        index = train_cols.index(name)
        for i in range(2):
            fpr[name, i], tpr[name, i], _ = roc_curve(
                labels[:, index] == i, preds[:, index]
            )
            roc_auc_scores[name, i] = roc_auc_score(
                labels[:, index] == i, preds[:, index]
            )

    return fpr, tpr, roc_auc_scores


def plot_roc_auc(fpr, tpr, roc_auc_scores, train_cols):
    plt.figure(figsize=(6, 6))
    lw = 2
    class_name = ["negative", "positive"]
    train_cols = [
        "Cardiomegaly",
        "Edema",
        "Consolidation",
        "Atelectasis",
        "Pleural Effusion",
    ]
    colors = ["aqua", "darkorange", "cornflowerblue", "purple", "magenta"]

    for name in train_cols:
        color = colors.pop(0)
        plt.plot(
            fpr[name, 1],
            tpr[name, 1],
            color=color,
            lw=lw,
            label="{0} ({1}) (area = {2:0.2f})".format(
                name, class_name[1], roc_auc_scores[name, 1]
            ),
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.title("Receiver operating characteristic curve", fontsize=12)
    plt.tight_layout()
    plt.show()


def cm_display(preds, labels, train_cols):
    preds = np.where(preds < 0.5, 0.0, preds)
    preds = np.where(preds >= 0.5, 1.0, preds)
    num_labels = labels.shape[1]
    figs = dict()
    figs_norm = dict()

    for i in range(num_labels):
        cm = confusion_matrix(labels[:, i], preds[:, i])
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        figs[i] = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["negative", "positive"]
        )
        figs_norm[i] = ConfusionMatrixDisplay(
            confusion_matrix=cm_norm, display_labels=["negative", "positive"]
        )

    for i in range(num_labels):
        figure = plt.figure(figsize=(3, 3))
        figs[i].plot()
        plt.title("Confusion Matrix - {0}".format(train_cols[i]), fontsize=12)
        figs_norm[i].plot()
        plt.title(
            "Normalized Confusion Matrix - {0}".format(train_cols[i]), fontsize=12
        )
        plt.show()


def reports(preds, labels, train_cols):
    preds = np.where(preds < 0.5, 0.0, preds)
    preds = np.where(preds >= 0.5, 1.0, preds)
    for i in range(len(train_cols)):
        label_name = train_cols[i]
        print("\n******************************************************")
        print(f"Classfication Reports {label_name}".center(54))
        print("******************************************************")
        print(
            classification_report(
                labels[:, i],
                preds[:, i],
                target_names=["negative", "positive"],
                zero_division=0,
            )
        )
        print("******************************************************")


def report_metrics(preds, labels, print_classification_result=True):
    train_cols = [
        "Cardiomegaly",
        "Edema",
        "Consolidation",
        "Atelectasis",
        "Pleural Effusion",
    ]

    # accuracy of each labels
    accuracy_list = get_accuracy_list(preds, labels)
    for i in range(len(train_cols)):
        print("---> [{0}] accuracy = {1}".format(train_cols[i], accuracy_list[i]))
    print("===> Total accuracy = {0} <===".format(np.mean(np.array(accuracy_list))))

    # for roc curve
    # fpr, tpr, roc_auc_scores = roc_curves(preds, labels, train_cols)
    # plot_roc_auc(fpr, tpr, roc_auc_scores, train_cols)

    # confusion matrix
    # cm_display(preds, labels, train_cols)

    # classification reports
    if print_classification_result:
        reports(preds, labels, train_cols)


class TestMetricsReporter:
    def __init__(self, hydra_cfg, preds, targets):
        self.preds = preds
        self.targets = targets
        self.hydra_cfg = hydra_cfg
        self.save_dir = os.path.join(hydra_cfg.log_dir, "images")


class AUROCMetricReporter(TestMetricsReporter):
    def __init__(self, hydra_cfg, preds, targets):
        super().__init__(hydra_cfg, preds, targets)

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
        target_columns = list(self.hydra_cfg.Dataset.train_cols)
        colors = ["orange", "green", "blue", "purple", "pink"]

        plt.clf()
        for idx in range(5):
            self.plot_class_auroc_details(
                self.targets[:, idx],
                self.preds[:, idx],
                col_name=target_columns[idx],
                overlap=True,
                color=colors[idx],
            )

        plt.title("All Classes' ROC CURVE", fontweight="bold")
        plt.savefig(os.path.join(self.save_dir, "overlap_roc_curve.png"), dpi=300)
