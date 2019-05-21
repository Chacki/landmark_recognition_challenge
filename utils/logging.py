from absl import flags
from ignite import metrics
from ignite.contrib import handlers
from ignite.engine import Events

FLAGS = flags.FLAGS


def attach_loggers(train_engine, eval_engine):
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
    tensorboard_logger.attach(
        eval_engine,
        log_handler=handlers.tensorboard_logger.OutputHandler(
            tag="validation", metric_names=["gap"], another_engine=train_engine
        ),
        event_name=Events.EPOCH_COMPLETED,
    )
