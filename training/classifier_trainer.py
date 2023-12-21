from typing import Literal, Optional

import lightning as L
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import ClasswiseWrapper
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    ConfusionMatrix,
    F1Score,
    MulticlassStatScores,
    Specificity,
    Precision,
)
from torchmetrics.utilities.compute import _adjust_weights_safe_divide, _safe_divide

CLASSES = ["NORMAL", "BENIGN", "MALIGNANT"]  # FIXME


def _sensitivity_reduce(
    tp: torch.Tensor,
    fp: torch.Tensor,
    tn: torch.Tensor,
    fn: torch.Tensor,
    average: Optional[Literal["binary", "micro", "macro", "weighted", "none"]],
    multidim_average: Literal["global", "samplewise"] = "global",
    multilabel: bool = False,
) -> torch.Tensor:
    if average == "binary":
        return _safe_divide(tn, tn + fp)
    if average == "micro":
        tn = tn.sum(dim=0 if multidim_average == "global" else 1)
        fp = fp.sum(dim=0 if multidim_average == "global" else 1)
        return _safe_divide(tn, tn + fp)

    specificity_score = _safe_divide(tp, tp + fn)
    return _adjust_weights_safe_divide(
        specificity_score, average, multilabel, tp, fp, fn
    )


class MulticlassSensitivity(MulticlassStatScores):
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Class"

    def compute(self) -> torch.Tensor:
        """Compute metric."""
        tp, fp, tn, fn = self._final_state()
        return _sensitivity_reduce(
            tp, fp, tn, fn, average=self.average, multidim_average=self.multidim_average
        )


class ClassifierTrainer(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        num_classes: int,
        classes: list[str] = CLASSES,
    ) -> None:
        super().__init__()

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.classes = classes

        self.net = net

        self.criterion = nn.CrossEntropyLoss()

        #          ClasswiseWrapper(
        # ...    MulticlassAccuracy(num_classes=3, average=None),
        # ...    labels=["horse", "fish", "dog"]
        # ... )

        self.train_metrics = {
            "acc_avg": Accuracy(task="multiclass", num_classes=num_classes),
            "acc": ClasswiseWrapper(
                Accuracy(task="multiclass", num_classes=num_classes, average=None),
                labels=classes,
                prefix="acc_",
                postfix="_train",
            ),
            "f1_avg": F1Score(task="multiclass", num_classes=num_classes),
            "f1": ClasswiseWrapper(
                F1Score(task="multiclass", num_classes=num_classes, average=None),
                labels=classes,
                prefix="f1_",
                postfix="_train",
            ),
            "sen_avg": MulticlassSensitivity(num_classes=num_classes),
            "sen": ClasswiseWrapper(
                MulticlassSensitivity(num_classes=num_classes, average=None),
                labels=classes,
                prefix="sen_",
                postfix="_train",
            ),
            "spe_avg": Specificity(task="multiclass", num_classes=num_classes),
            "spe": ClasswiseWrapper(
                Specificity(task="multiclass", num_classes=num_classes, average=None),
                labels=classes,
                prefix="spe_",
                postfix="_train",
            ),
            "pre_avg": Precision(task="multiclass", num_classes=num_classes),
            "pre": ClasswiseWrapper(
                Precision(task="multiclass", num_classes=num_classes, average=None),
                labels=classes,
                prefix="pre_",
                postfix="_train",
            ),
        }

        self.val_metrics = {
            "acc_avg": Accuracy(task="multiclass", num_classes=num_classes),
            "acc": ClasswiseWrapper(
                Accuracy(task="multiclass", num_classes=num_classes, average=None),
                labels=classes,
                prefix="acc_",
                postfix="_val",
            ),
            "f1_avg": F1Score(task="multiclass", num_classes=num_classes),
            "f1": ClasswiseWrapper(
                F1Score(task="multiclass", num_classes=num_classes, average=None),
                labels=classes,
                prefix="f1_",
                postfix="_val",
            ),
            "sen_avg": MulticlassSensitivity(num_classes=num_classes),
            "sen": ClasswiseWrapper(
                MulticlassSensitivity(num_classes=num_classes, average=None),
                labels=classes,
                prefix="sen_",
                postfix="_val",
            ),
            "spe_avg": Specificity(task="multiclass", num_classes=num_classes),
            "spe": ClasswiseWrapper(
                Specificity(task="multiclass", num_classes=num_classes, average=None),
                labels=classes,
                prefix="spe_",
                postfix="_val",
            ),
            "pre_avg": Precision(task="multiclass", num_classes=num_classes),
            "pre": ClasswiseWrapper(
                Precision(task="multiclass", num_classes=num_classes, average=None),
                labels=classes,
                prefix="pre_",
                postfix="_val",
            ),
        }

        self.roc_auc_avg_train = AUROC(task="multiclass", num_classes=num_classes)
        self.roc_auc_avg_val = AUROC(task="multiclass", num_classes=num_classes)

        self.roc_auc_train = ClasswiseWrapper(
            AUROC(task="multiclass", num_classes=num_classes, average=None),
            labels=classes,
            prefix="auc_",
            postfix="_train",
        )
        self.roc_auc_val = ClasswiseWrapper(
            AUROC(task="multiclass", num_classes=num_classes, average=None),
            labels=classes,
            prefix="auc_",
            postfix="_val",
        )

        self.cm_train = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.cm_val = ConfusionMatrix(task="multiclass", num_classes=num_classes)

        self.save_hyperparameters(ignore=["net"])

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer

        return [self.optimizer], [self.scheduler]

    def step(self, batch, batch_idx, is_training):
        images, labels = batch

        out_class_logits = self.net(images)

        loss = self.criterion(out_class_logits, labels)

        predicted_probs = torch.softmax(out_class_logits, 1)
        predicted = torch.argmax(predicted_probs, dim=1)

        if is_training:
            self.log("running_loss_train", loss.item(), on_epoch=False, on_step=True)
            self.log("loss_train", loss.item(), on_epoch=True, on_step=False)

            for metric in self.train_metrics.values():
                metric.to(self.device).update(predicted, labels)

            self.roc_auc_train.update(predicted_probs, labels)
            self.roc_auc_avg_train.update(predicted_probs, labels)
            self.cm_train.update(predicted, labels)

            self.log(
                "roc_auc_avg_train",
                self.roc_auc_avg_train,
                on_epoch=True,
                on_step=False,
            )

        else:
            self.log("running_loss_val", loss.item(), on_epoch=False, on_step=True)
            self.log("loss_val", loss.item(), on_epoch=True, on_step=False)

            for metric in self.val_metrics.values():
                metric.to(self.device).update(predicted, labels)

            self.roc_auc_val.update(predicted_probs, labels)
            self.roc_auc_avg_val.update(predicted_probs, labels)
            self.cm_val.update(predicted, labels)

            self.log(
                "roc_auc_avg_val", self.roc_auc_avg_val, on_epoch=True, on_step=False
            )

        return loss

    def on_train_epoch_end(self):
        fig, ax = self.cm_train.plot(labels=self.classes)

        self.logger.log_image(key="cm_train", images=[fig])

        plt.close(fig)

        self.log_dict(self.roc_auc_train.compute(), on_epoch=True, on_step=False)
        for name, metric in self.train_metrics.items():
            if isinstance(metric, ClasswiseWrapper):
                self.log_dict(metric.compute())
            else:
                self.log(
                    name + "_train", metric.compute(), on_epoch=True, on_step=False
                )
            metric.reset()

    def on_validation_epoch_end(self):
        fig, ax = self.cm_val.plot(labels=self.classes)

        self.logger.log_image(key="cm_val", images=[fig])

        plt.close(fig)

        self.log_dict(self.roc_auc_val.compute(), on_epoch=True, on_step=False)

        for name, metric in self.val_metrics.items():
            if isinstance(metric, ClasswiseWrapper):
                self.log_dict(metric.compute())
            else:
                self.log(name + "_val", metric.compute(), on_epoch=True, on_step=False)
            metric.reset()

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, is_training=True)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, is_training=False)
