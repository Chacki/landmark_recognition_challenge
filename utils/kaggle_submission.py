from os import path

import pandas as pd
import torch

# from absl import flags
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import landmark_recognition

from . import evaluation

# flags.DEFINE_integer("num_top_predicts", 100, "Number of Predictions")
# FLAGS = flags.FLAGS
class Flags:
    def __init__(self):
        self.batch_size = 256
        self.device = "cuda"
        self.num_top_predicts = 100
        self.dataset = "google-landmark"
        self.evaluation_dir = "evaluation"


FLAGS = Flags()


def generate_submission(
    model,
    gallery_dl=None,
    gallery_feats=None,
    gallery_labels=None,
    label_transforms=None,
):
    topk = FLAGS.num_top_predicts
    model.eval()
    model.to(FLAGS.device)
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
    if gallery_dl is not None or gallery_feats is not None:
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
    else:
        classes = []
        final_confidence = []
        with torch.no_grad():
            for img, _ in query_dl:
                out = torch.max(model(img.to(FLAGS.device)), dim=1)
                classes.append(out.indices.cpu())
                final_confidence.append(out.values.cpu())
        classes = torch.cat(classes, dim=0)
        final_confidence = torch.cat(final_confidence, dim=0)
        final_confidence.sub_(final_confidence.min())
        final_confidence.div_(final_confidence.max())
        final_confidence = final_confidence.numpy()
        final_labels = label_transforms(classes.numpy())
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
