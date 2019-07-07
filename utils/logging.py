import torch
from absl import flags
from ignite import metrics
from ignite.contrib import handlers
from ignite.contrib.handlers.base_logger import BaseHandler
from ignite.engine import Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from tqdm.auto import tqdm

FLAGS = flags.FLAGS


class EmbeddingHandler(BaseHandler):
    """ Image handler for tensorboard.
    Generate images after given number of iterations and add generated images
    to tensorboard.
    """

    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

    def __call__(self, engine, logger, event_name):
        with torch.no_grad():
            embeddings = []
            images = []
            labels = []
            for img, label in tqdm(self.dataloader):
                embeddings.append(
                    torch.nn.functional.normalize(self.model(img.cuda())).cpu()
                )
                images.append(img)
                labels.append(label)
            embeddings = torch.cat(embeddings, 0)
            images = torch.cat(images, 0)
            labels = torch.cat(labels, 0)
            # 1200 because of sprite image limits of tensorboard
            logger.writer.add_embedding(
                embeddings[:1200],
                labels[:1200],
                images[:1200],
                engine.state.epoch,
            )


def attach_loggers(
    train_engine,
    eval_engine,
    model,
    early_stopping_metric=None,
    additional_tb_log_handler=[],
):
    metrics.RunningAverage(output_transform=lambda x: x).attach(
        train_engine, "loss"
    )
    handlers.ProgressBar(persist=True).attach(train_engine, ["loss"])
    tensorboard_logger = handlers.TensorboardLogger(FLAGS.tensorboard_dir)
    tensorboard_logger.attach(
        train_engine,
        log_handler=handlers.tensorboard_logger.OutputHandler(
            tag="training", output_transform=lambda loss: {"loss": loss}
        ),
        event_name=Events.ITERATION_COMPLETED,
    )
    for log_handler, event in additional_tb_log_handler:
        tensorboard_logger.attach(
            train_engine, log_handler=log_handler, event_name=event
        )
    if eval_engine is not None:
        handlers.ProgressBar(persist=True).attach(eval_engine)
        tensorboard_logger.attach(
            eval_engine,
            log_handler=handlers.tensorboard_logger.OutputHandler(
                tag="validation",
                output_transform=lambda engine: engine.state.metrics,
                another_engine=train_engine,
            ),
            event_name=Events.EPOCH_COMPLETED,
        )
        # early stopping and checkpoint
        eval_engine.add_event_handler(
            Events.COMPLETED,
            ModelCheckpoint(
                FLAGS.checkpoint_dir,
                "model",
                n_saved=10,
                score_function=lambda x: x.state.metrics["gap"],
                score_name="gap",
                require_empty=False,
            ),
            {"model": model},
        )
        if early_stopping_metric is not None:
            eval_engine.add_event_handler(
                Events.COMPLETED,
                EarlyStopping(
                    10,
                    score_function=lambda x: x.state.metrics[
                        early_stopping_metric
                    ],
                    trainer=train_engine,
                ),
            )
    else:
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            ModelCheckpoint(
                FLAGS.checkpoint_dir,
                "state_dict",
                save_interval=1,
                n_saved=10,
                require_empty=False,
            ),
            {"model": model},
        )
