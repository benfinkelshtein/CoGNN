from enum import Enum, auto
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, L1Loss
import torch
from typing import NamedTuple, List
from torchmetrics import Accuracy, AUROC, MeanAbsoluteError, MeanSquaredError, F1Score, AveragePrecision
import math
from torch_geometric.data import Data
import numpy as np


class LossesAndMetrics(NamedTuple):
    train_loss: float
    val_loss: float
    test_loss: float
    train_metric: float
    val_metric: float
    test_metric: float

    def get_fold_metrics(self):
        return torch.tensor([self.train_metric, self.val_metric, self.test_metric])


class MetricType(Enum):
    """
        an object for the different metrics
    """
    # classification
    ACCURACY = auto()
    MULTI_LABEL_AP = auto()
    AUC_ROC = auto()

    # regression
    MSE_MAE = auto()

    def apply_metric(self, scores: np.ndarray, target: np.ndarray) -> float:
        if isinstance(scores, np.ndarray):
            scores = torch.from_numpy(scores)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        num_classes = scores.size(1)  # target.max().item() + 1
        if self is MetricType.ACCURACY:
            metric = Accuracy(task="multiclass", num_classes=num_classes)
        elif self is MetricType.MULTI_LABEL_AP:
            metric = AveragePrecision(task="multilabel", num_labels=num_classes).to(scores.device)
            result = metric(scores, target.int())
            return result.item()
        elif self is MetricType.MSE_MAE:
            metric = MeanAbsoluteError()
        elif self is MetricType.AUC_ROC:
            metric = AUROC(task="multiclass", num_classes=num_classes)
        else:
            raise ValueError(f'MetricType {self.name} not supported')

        metric = metric.to(scores.device)
        result = metric(scores, target)
        return result.item()

    def is_classification(self) -> bool:
        if self in [MetricType.AUC_ROC, MetricType.ACCURACY, MetricType.MULTI_LABEL_AP]:
            return True
        elif self is MetricType.MSE_MAE:
            return False
        else:
            raise ValueError(f'MetricType {self.name} not supported')

    def is_multilabel(self) -> bool:
        return self is MetricType.MULTI_LABEL_AP

    def get_task_loss(self):
        if self.is_classification():
            if self.is_multilabel():
                return BCEWithLogitsLoss()
            else:
                return CrossEntropyLoss()
        elif self is MetricType.MSE_MAE:
            return MSELoss()
        else:
            raise ValueError(f'MetricType {self.name} not supported')

    def get_out_dim(self, dataset: List[Data]) -> int:
        if self.is_classification():
            if self.is_multilabel():
                return dataset[0].y.shape[1]
            else:
                return int(max([data.y.max().item() for data in dataset]) + 1)
        else:
            return dataset[0].y.shape[-1]

    def higher_is_better(self):
        return self.is_classification()

    def src_better_than_other(self, src: float, other: float) -> bool:
        if self.higher_is_better():
            return src > other
        else:
            return src < other

    def get_worst_losses_n_metrics(self) -> LossesAndMetrics:
        if self.is_classification():
            return LossesAndMetrics(train_loss=math.inf, val_loss=math.inf, test_loss=math.inf,
                                    train_metric=-math.inf, val_metric=-math.inf, test_metric=-math.inf)
        else:
            return LossesAndMetrics(train_loss=math.inf, val_loss=math.inf, test_loss=math.inf,
                                    train_metric=math.inf, val_metric=math.inf, test_metric=math.inf)
