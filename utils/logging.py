from absl import flags
from ignite import metrics
from ignite.contrib import handlers
from ignite.engine import Events
from ignite.handlers import EarlyStopping, ModelCheckpoint

FLAGS = flags.FLAGS


def attach_loggers(train_engine, eval_engine, model):
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
    if eval_engine is not None:
        handlers.ProgressBar(persist=True).attach(eval_engine)
        tensorboard_logger.attach(
            eval_engine,
            log_handler=handlers.tensorboard_logger.OutputHandler(
                tag="validation", metric_names=["gap"], another_engine=train_engine
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
        eval_engine.add_event_handler(
            Events.COMPLETED,
            EarlyStopping(
                10,
                score_function=lambda x: x.state.metrics["gap"],
                trainer=train_engine,
            ),
        )
    else:
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            ModelCheckpoint(
                FLAGS.checkpoint_dir,
                "model",
                n_saved=10,
                score_function=lambda x: x.state.metrics["loss"],
                score_name="loss",
                require_empty=False,
            ),
            {"model": model},
        )
