"""
File: exp_08_auxiliary_losses.py
Description: experiment with ensemble learning.
             This experiment does NOT use the style of other experiments in
             this repository.
"""

from pytorch_lightning import Trainer, LightningModule
from test_tube import Experiment, HyperOptArgumentParser
from efficientnet_pytorch import EfficientNet
from torch import nn
import torch
from loss.triplet_loss import OnlineHardMining
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils import data
from os import path
import numpy as np
from utils.finch import FINCH
import os


class ClusteredDataset(data.Dataset):
    def __init__(self, dataframe, directory, transform):
        super().__init__()
        self.directory = directory
        self.transform = transform
        self.ids = dataframe["id"].to_numpy()
        self.labels = dataframe["landmark_id"].to_numpy()
        self.model = model
        self.clusters = []

    def __getitem__(self, index):
        img_name, label, cluster = (
            self.ids[index],
            self.labels[index],
            self.clusters[index],
        )
        img_path = path.join(self.directory, img_name + ".jpg")
        return (
            self.transform(img_path),
            torch.from_numpy(np.array(label)),
            torch.from_numpy(np.array(cluster)),
        )

    def __len__(self):
        return self.ids.shape[0]


def build_model(embedding_dim):
    model = EfficientNet.from_pretrained(
        "efficientnet-b3", num_classes=embedding_dim
    )
    return model


class Model(LightningModule):
    def __init__(self, num_clusters, embedding_dim, margin):
        super().__init__()
        self.num_clusters = num_clusters

        self.feature_extractors = nn.ModuleList(
            [build_model(embedding_dim) for i in range(num_clusters)]
        )
        test_out = self.feature_extractors[0](torch.rand(1, 3, 224, 224))

        self.gating_feats = build_model(num_clusters)
        self.gating_clf = nn.Linear(test_out.size(-1), num_clusters)
        self.loss = OnlineHardMining(margin)

        transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: Image.open(x).convert("RGB")),
                transforms.RandomCrop(224, pad_if_needed=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )
        self.train_ds = ClusteredDataset(
            pd.read_csv("./data/google-landmark/valid_train.csv"),
            "./data/google-landmark/train",
            transform,
        )

    def training_step(self, batch, batch_nb):
        img, label, cluster = batch

        gating_feats = self.gating_feats(img)
        gate = nn.functional.softmax(self.gating_clf(gating_feats))

        img_feats = torch.stack(
            [
                nn.functional.normalize(self.embed(model(img)))
                for model in self.feature_extractors
            ],
            1,
        )
        weighted_img_feats = img_feats * gate
        final_img_feat = nn.functional.normalize(
            torch.max(weighted_img_feats, 1)
        )

        loss_embed = self.loss(final_img_feat, label)
        loss_gating = nn.functional.cross_entropy(gate, cluster)
        return {
            "loss": loss_embed + loss_gating,
            "pgbar": {"loss_embed": loss_embed, "loss_gating": loss_gating},
        }

    def validation_step(self, batch, batch_nb):
        img, label, cluster = batch
        feats = self.gating_feats(img)
        return {"feats": feats}

    def validation_end(self, outputs):
        feats = torch.cat((x["feats"] for x in outputs))
        cluster, num_cluster, req_cluster = FINCH(
            feats, req_clust=self.num_clusters
        )
        self.train_ds.clusters = req_cluster

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters())
        return [optim]

    def tng_dataloader(self):
        train_dl = data.DataLoader(
            dataset=self.train_ds, batch_size=32, num_workers=8
        )
        return train_dl

    def val_dataloader(self):
        valid_dl = data.DataLoader(
            dataset=self.train_ds, batch_size=32, num_workers=8
        )
        return valid_dl


def main():
    parser = HyperOptArgumentParser()
    parser.opt_list(
        "--num_cluster", default=2, type=int, tunable=True, options=[2, 3, 4]
    )
    parser.opt_list(
        "--embedding_dim",
        default=512,
        type=int,
        tunable=True,
        options=[256, 512, 1024],
    )
    parser.opt_range(
        "--margin",
        default=0.5,
        type=float,
        tunable=True,
        low=0.2,
        high=1.0,
        nb_samples=8,
    )
    hparams = parser.parse_args()

    experiment = Experiment(save_dir=os.path.join(os.getcwd(), "bla"))
    experiment.argparse(hparams)
    experiment.save()
    trainer = Trainer(experiment=experiment)
    model = Model(hparams.num_cluster, hparams.embedding_dim, hparams.margin)
    trainer.fit(model)


if __name__ == "__main__":
    main()
