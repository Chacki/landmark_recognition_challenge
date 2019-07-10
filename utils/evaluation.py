import itertools
from operator import itemgetter

import ignite
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import pairwise_distances
from torch import nn
from tqdm.auto import tqdm


class CalculateAccuracy:
    def __init__(self, gallery_dl, model):
        self.gallery_dl = gallery_dl
        self.model = model

    def __call__(self, predictions, targets):
        labels, dists = get_topk(self.model, predictions, self.gallery_dl, 1)
        acc = (
            torch.tensor(labels[:, 0]).type_as(targets) == targets
        ).sum() / targets.size(0)
        return acc


def extract_feats(dataloader, model, idx):
    feats = []
    with torch.no_grad():
        for img in tqdm(map(itemgetter(idx), dataloader), "Extract feats"):
            feat = nn.functional.normalize(
                model(img.cuda()).view(img.size(0), -1)
            ).cpu()
            feats.append(feat)
        feats = torch.cat(feats, axis=0)
    return feats


def get_topk(model: nn.Module, query_feats: torch.Tensor, gallery_dl, topk):
    topk_dists = query_feats.new_zeros((query_feats.size(0), topk)) + 1
    topk_idxs = query_feats.new_zeros((query_feats.size(0), topk))
    topk_labels = query_feats.new_zeros((query_feats.size(0), topk)) - 1
    nn.functional.normalize(query_feats, out=query_feats)
    with torch.no_grad():
        for img, label in tqdm(gallery_dl):
            feats = nn.functional.normalize(model(img.to(query_feats.device)))
            dists = query_feats.matmul(feats.t())
            tmp_dists = torch.cat((topk_dists, dists), dim=1)
            tmp_labels = torch.cat(
                (
                    topk_labels,
                    label.unsqueeze(0).expand(topk_labels.size(0), -1),
                ),
                dim=1,
            )
            tmp_dists.topk(
                topk, dim=1, sorted=False, out=(topk_dists, topk_idxs)
            )
            tmp_labels.gather(1, topk_idxs, out=topk_labels)
    return topk_labels, topk_dists


def predict_labels(
    topk,
    model=None,
    query_dl=None,
    query_feats=None,
    gallery_dl=None,
    gallery_feats=None,
    gallery_labels=None,
):
    assert (query_dl is None) != (query_feats is None)
    assert (gallery_feats is None) == (gallery_labels is None)
    assert (gallery_dl is None) != (gallery_feats is None)
    if query_feats is None:
        query_feats = extract_feats(query_dl, model, 0)
    if gallery_dl is not None:
        topk_labels, topk_dists = get_topk(model, query_feats, gallery_dl, topk)
        topk_dists, topk_idxs = topk_dists.sort(dim=1, descending=True)
        topk_labels.gather(1, topk_idxs, out=topk_labels)
    else:
        query_feats = nn.functional.normalize(query_feats)
        gallery_feats = nn.functional.normalize(gallery_feats)
        dists = query_feats.matmul(gallery_feats.t())
        exp_l = gallery_labels.unsqueeze(0).expand(query_feats.size(0), -1)
        topk_dists, topk_idxs = dists.topk(topk, dim=1, sorted=True)
        topk_labels = exp_l.gather(1, topk_idxs)
    return topk_labels, topk_dists


def gap_score(pred, conf, true, return_x=False):
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
        self.clf = pyflann.FLANN()
        self.engine = ignite.engine.create_supervised_evaluator(
            self.model,
            metrics={"gap": ignite.metrics.EpochMetric(self.func)},
            device=FLAGS.device,
            non_blocking=True,
        )
        self.db_target = None
        self.params = None

    def calc(self, db_feats, db_target):
        self.db_target = db_target.cpu().numpy()
        self.params = self.clf.build_index(
            db_feats.cpu().numpy(), algorithm="autotuned", target_precision=0.9
        )
        self.engine.run(self.test_loader)
        gap = self.engine.state.metrics["gap"]
        # TODO dont use nested evaluaters. Problem with nested no_grad env
        torch.set_grad_enabled(True)
        return gap

    def func(self, test_feats, test_target):
        num_neighbors = 5
        result, distance = self.clf.nn_index(
            test_feats.cpu().numpy(),
            num_neighbors=num_neighbors,
            checks=self.params["checks"],
        )
        nearest_targets = self.db_target[result]
        pred, count = np.unique(nearest_targets, return_counts=True, axis=-1)
        pred = pred[np.arange(len(test_target)), np.argmax(count, axis=-1)]
        distance = np.ma.array(
            distance, mask=(nearest_targets.T == pred).T
        ).mean(axis=-1)
        conf = 1 - distance / np.max(distance)
        conf *= np.max(count, axis=-1) / num_neighbors
        gap = gap_score(pred, conf, test_target.cpu().numpy())
        return gap


def build_evaluater(model, test_loader):
    gapmetric = GAPMetric(model, test_loader)
    evaluater = ignite.engine.create_supervised_evaluator(
        model,
        metrics={"gap": ignite.metrics.EpochMetric(gapmetric.calc)},
        device=FLAGS.device,
        non_blocking=True,
    )
    return evaluater


def attach_eval(evaluater, trainer, db_loader):
    def _eval(engine):
        if engine.state.epoch % FLAGS.eval_epochs == 0:
            evaluater.run(db_loader)
            print("GAP:", evaluater.state.metrics["gap"])

    trainer.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, _eval)
