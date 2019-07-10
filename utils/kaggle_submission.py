from os import path

import pandas as pd
from absl import flags
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import landmark_recognition

from . import evaluation

flags.DEFINE_integer("num_top_predicts", 100, "Number of Predictions")
FLAGS = flags.FLAGS


def generate_submission(
    model, gallery_dl=None, gallery_feats=None, gallery_labels=None
):
    topk = FLAGS.num_top_predicts
    model.eval()
    csv_path = path.join(
        "./data/", FLAGS.dataset, "recognition_sample_submission.csv"
    )
    query_df = pd.read_csv(csv_path)
    query_df["landmark_id"] = 0
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: Image.open(x).convert("RGB")),
            transforms.RandomCrop(224, pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    query_ds = landmark_recognition.Dataset(
        query_df, path.join(path.dirname(csv_path), "test"), transform
    )
    query_dl = DataLoader(
        query_ds, batch_size=FLAGS.batch_size, num_workers=16, shuffle=False
    )
    labels, _ = evaluation.predict_labels(
        topk,
        model,
        query_dl=query_dl,
        gallery_dl=gallery_dl,
        gallery_feats=gallery_feats,
        gallery_labels=gallery_labels,
    )
    final_labels = labels[:, 0].cpu().numpy()
    final_confidence = (
        ((labels == labels[:, 0].unsqueeze(-1)).sum(-1).float() / topk)
        .cpu()
        .numpy()
    )
    query_df["labels"] = final_labels
    query_df["confidence"] = final_confidence
    query_df["landmarks"] = query_df["labels"].map(
        lambda x: str(x) + " "
    ) + query_df["confidence"].map(str)
    query_df.to_csv(
        path.join(FLAGS.evaluation_dir, "kaggle_submission.csv"),
        columns=["id", "landmarks"],
        index=False,
    )
