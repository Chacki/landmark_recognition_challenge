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
from loss.triplet_loss import OnlineHardMining, TripletLoss
from loss.center_loss import CenterLoss
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils import data
from os import path
import numpy as np
from utils.finch import FINCH
import os
from dataset.sampler import RandomIdentitySampler
from utils.data import LabelEncoder
from pytorch_lightning.callbacks import ModelCheckpoint

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
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

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
        # self.clusters = np.random.randint(
        #     low=0, high=self.num_clusters, size=self.labels.shape
        # )
        # return
        test_out = self.model(torch.rand(1, 3, 224, 224).cuda())
        batch_size = dataloader.batch_size
        with torch.no_grad():
            self.model.eval()
            feats = [
                self.model(batch[0].cuda()).cpu() for batch in tqdm(dataloader)
            ]
            feats = torch.cat(feats, dim=0)
            cluster, num_cluster, req_cluster = FINCH(
                feats.numpy(), req_clust=self.num_clusters
            )
            self.model.train()
        self.clusters = req_cluster.astype(np.long)


def build_model(embedding_dim, feats_only=False):
    model = EfficientNet.from_pretrained(
        "efficientnet-b2", num_classes=embedding_dim
    )
    if feats_only:
        model._fc = nn.Flatten()
    return model


class Model(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        num_clusters = hparams.num_cluster
        embedding_dim = hparams.embedding_dim
        margin = hparams.margin
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

        # self.loss = OnlineHardMining(margin)
        self.loss = TripletLoss(margin=margin)

        transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: Image.open(x).convert("RGB")),
                transforms.RandomCrop(224, pad_if_needed=True),
                transforms.RandomRotation(90),
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
        print(len(np.unique(self.train_ds.labels)))
        self.center_loss = CenterLoss(
            num_classes=len(np.unique(self.train_ds.labels)),
            feature_dim=embedding_dim,
            sparse=False,
        )

    def forward(self, img):
        gate = self.gating_model(img)

        img_feats = gate.new_empty(
            img.size(0), self.num_clusters, self.embedding_dim
        )
        for idx, model in enumerate(self.feature_extractors):
            img_feats[:, idx, :] = nn.functional.normalize(model(img))

        weighted_img_feats = img_feats * gate.unsqueeze(-1)
        final_img_feat = weighted_img_feats.sum(dim=1)
        return final_img_feat

    def training_step(self, batch, batch_nb):
        img, label, cluster = batch

        gate = self.gating_model(img)

        img_feats = gate.new_empty(
            img.size(0), self.num_clusters, self.embedding_dim
        )
        for idx, model in enumerate(self.feature_extractors):
            img_feats[:, idx, :] = nn.functional.normalize(model(img))

        weighted_img_feats = img_feats * gate.unsqueeze(-1)
        final_img_feat = weighted_img_feats.sum(dim=1)

        loss_triplet = self.loss(final_img_feat, label)
        # loss_center = self.center_loss(final_img_feat, label)
        loss_gating = nn.functional.cross_entropy(gate, cluster)
        acc_gating = torch.argmax(gate, -1).eq(cluster).float().mean()
        return {
            "loss": loss_triplet + loss_gating,
            "prog": {
                "loss_triplet": loss_triplet,
                # "loss_center": loss_center,
                "loss_gating": loss_gating,
                "acc_gating": acc_gating,
            },
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


from utils.kaggle_submission import generate_submission


def test(hparams):
    model = Model(hparams)
    model.load_from_metrics(
        weights_path="model/weights.ckpt/_ckpt_epoch_5.ckpt",
        tags_csv=f"bla/default/version_{hparams.version}/meta_tags.csv",
        on_gpu=True,
        map_location=None,
    )

    model.train_ds.labels = model.train_ds.label_encoder.inverse_transform(
        model.train_ds.labels
    )
    model.freeze()
    model.eval()
    with torch.no_grad():
        generate_submission(
            model=model,
            gallery_dl=map(lambda x: (x[0], x[1]), model.val_dataloader),
            # gallery_labels=model.train_ds.label_encoder.inverse_transform(
            #     model.train_ds.labels
            # ),
        )


def main():
    parser = HyperOptArgumentParser()
    parser.opt_list(
        "--num_cluster", default=3, type=int, tunable=True, options=[3, 4, 5]
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
        default=1.0,
        type=float,
        tunable=True,
        low=0.2,
        high=1.0,
        nb_samples=8,
    )
    parser.add_argument("--version", default=None, type=int)
    parser.add_argument("--test", default=False, type=bool)
    hparams = parser.parse_args()
    if hparams.test:
        test(hparams)
        return

    experiment = Experiment(
        save_dir=os.path.join(os.getcwd(), "bla"), version=hparams.version
    )
    experiment.argparse(hparams)
    experiment.save()
    checkpoint_callback = ModelCheckpoint(
        "model/weights.ckpt",
        monitor="loss_triplet",
        mode="min",
        save_best_only=True,
    )
    trainer = Trainer(
        experiment=experiment, gpus=[0], checkpoint_callback=checkpoint_callback
    )
    model = Model(hparams)
    trainer.fit(model)


if __name__ == "__main__":
    main()
