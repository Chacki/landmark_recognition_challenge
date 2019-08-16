"""
File: exp_08_auxiliary_losses.py
Description: experiment with ensemble learning.
             This experiment does NOT use the style of other experiments in
             this repository.
"""

from pytorch_lightning import Trainer, LightningModule, data_loader
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
from dataset.sampler import RandomIdentitySampler

from tqdm.auto import tqdm


class ClusteredDataset(data.Dataset):
    def __init__(self, dataframe, directory, transform, num_clusters, model):
        super().__init__()
        self.directory = directory
        self.transform = transform
        self.model = model
        self.num_clusters = num_clusters
        self.ids = dataframe["id"].to_numpy()
        self.labels = dataframe["landmark_id"].to_numpy()
        self.clusters = np.random.randint(
            0, high=num_clusters, size=self.labels.shape
        )

    def __getitem__(self, index):
        img_name, label, cluster = (
            self.ids[index],
            self.labels[index],
            self.clusters[index],
        )
        img_path = path.join(self.directory, img_name + ".jpg")
        return (
            self.transform(img_path),
            torch.from_numpy(np.array(label, dtype=np.long)),
            torch.from_numpy(np.array(cluster, dtype=np.long)),
        )

    def __len__(self):
        return self.ids.shape[0]

    def update_clusters(self, dataloader):
        test_out = self.model(torch.rand(1, 3, 224, 224).cuda())
        batch_size = dataloader.batch_size
        with torch.no_grad():
            feats = [
                self.model(batch[0].cuda()).cpu() for batch in tqdm(dataloader)
            ]
            # feats = torch.empty(
            #     (self.labels.shape[0], test_out.size(1)), dtype=torch.float32
            # )
            # l = 0
            # u = batch_size
            # for idx, batch in enumerate(tqdm(dataloader)):
            #     feats[l:u] = self.model(batch[0].cuda()).cpu()
            #     l = u
            #     u += batch_size
            feats = torch.cat(feats, dim=0)
            cluster, num_cluster, req_cluster = FINCH(
                feats.numpy(), req_clust=self.num_clusters
            )
        self.clusters = req_cluster.astype(np.long)


def build_model(embedding_dim, feats_only=False):
    model = EfficientNet.from_pretrained(
        "efficientnet-b0", num_classes=embedding_dim
    )
    if feats_only:
        model._fc = nn.Flatten()
    return model


class Model(LightningModule):
    def __init__(self, num_clusters, embedding_dim, margin):
        super().__init__()
        self.num_clusters = num_clusters
        self.embedding_dim = embedding_dim

        self.feature_extractors = nn.ModuleList(
            [build_model(embedding_dim) for i in range(num_clusters)]
        )

        gating_feats = build_model(num_clusters, feats_only=True)
        test_out = gating_feats(torch.rand(1, 3, 224, 224))
        gating_clf = nn.Linear(test_out.size(-1), num_clusters)
        self.gating_model = nn.Sequential(
            gating_feats, gating_clf, nn.Softmax(dim=1)
        )

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
            num_clusters,
            gating_feats,
        )

    def training_step(self, batch, batch_nb):
        img, label, cluster = batch

        gate = self.gating_model(img)

        img_feats = gate.new_empty(
            img.size(0), self.num_clusters, self.embedding_dim
        )
        for idx, model in enumerate(self.feature_extractors):
            img_feats[:, idx, :] = nn.functional.normalize(model(img))

        weighted_img_feats = img_feats * gate.unsqueeze(-1)
        final_img_feat = nn.functional.normalize(
            torch.max(weighted_img_feats, 1).values
        )

        loss_embed = self.loss(final_img_feat, label)
        loss_gating = nn.functional.cross_entropy(gate, cluster)
        return {
            "loss": loss_embed + loss_gating,
            "prog": {"loss_embed": loss_embed, "loss_gating": loss_gating},
        }

    def on_epoch_start(self):
        self.train_ds.update_clusters(self.val_dataloader)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters())
        return [optim]

    @data_loader
    def tng_dataloader(self):
        train_dl = data.DataLoader(
            dataset=self.train_ds,
            batch_size=32,
            num_workers=8,
            sampler=RandomIdentitySampler(self.train_ds.labels),
        )
        return train_dl

    @data_loader
    def val_dataloader(self):
        valid_dl = data.DataLoader(
            dataset=self.train_ds, batch_size=256, num_workers=8
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
    trainer = Trainer(experiment=experiment, gpus=[0])
    model = Model(hparams.num_cluster, hparams.embedding_dim, hparams.margin)
    trainer.fit(model)


if __name__ == "__main__":
    main()
