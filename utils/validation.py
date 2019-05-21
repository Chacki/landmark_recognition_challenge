import traceback

import ignite
import numpy as np
import pandas as pd
import torch
from absl import flags
from sklearn.neighbors import KNeighborsClassifier

FLAGS = flags.FLAGS


def GAP_vector(pred, conf, true, return_x=False):
    """ https://www.kaggle.com/davidthaler/gap-metric
    Compute Global Average Precision (aka micro AP), the metric for the
    Google Landmark Recognition competition.
    This function takes predictions, labels and confidence scores as vectors.
    In both predictions and ground-truth, use None/np.nan for "no label".

    Args:
        pred: vector of integer-coded predictions
        conf: vector of probability or confidence scores for pred
        true: vector of integer-coded labels for ground truth
        return_x: also return the data frame used in the calculation

    Returns:
        GAP score
    """
    x = pd.DataFrame({"pred": pred, "conf": conf, "true": true})
    x.sort_values("conf", ascending=False, inplace=True, na_position="last")
    x["correct"] = (x.true == x.pred).astype(int)
    x["prec_k"] = x.correct.cumsum() / (np.arange(len(x)) + 1)
    x["term"] = x.prec_k * x.correct
    gap = x.term.sum() / x.true.count()
    if return_x:
        return gap, x
    else:
        return gap


class GAPMetric:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.knn = KNeighborsClassifier(n_jobs=-1)
        self.engine = ignite.engine.create_supervised_evaluator(
            self.model,
            metrics={"gap": ignite.metrics.EpochMetric(self.func)},
            device=FLAGS.device,
            non_blocking=True,
        )

    def calc(self, feats, target):
        self.knn.fit(feats.cpu().numpy(), target.cpu().numpy())
        self.engine.run(self.test_loader)
        gap = self.engine.state.metrics["gap"]
        # TODO dont use nested evaluaters. Problem with nested no_grad env
        torch.set_grad_enabled(True)
        return gap

    def func(self, test_feats, test_target):
        knn_proba = self.knn.predict_proba(test_feats.cpu().numpy())
        knn_idxs = np.argmax(knn_proba, axis=-1)
        conf = knn_proba[(np.arange(len(knn_idxs)), knn_idxs)]
        pred = np.vectorize(lambda x: self.knn.classes_[x])(knn_idxs)
        gap = GAP_vector(pred, conf, test_target.cpu().numpy())
        return gap


def build_validator(model, test_loader):
    gapmetric = GAPMetric(model, test_loader)
    evaluater = ignite.engine.create_supervised_evaluator(
        model,
        metrics={"gap": ignite.metrics.EpochMetric(gapmetric.calc)},
        device=FLAGS.device,
        non_blocking=True,
    )
    return evaluater
