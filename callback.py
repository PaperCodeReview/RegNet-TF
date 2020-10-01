import os
import yaml

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class OptionalLearningRateSchedule(LearningRateSchedule):
    def __init__(self, args, steps_per_epoch, initial_epoch):
        super(OptionalLearningRateSchedule, self).__init__()
        self.args = args
        self.steps_per_epoch = steps_per_epoch
        self.initial_epoch = initial_epoch
        self.lr_scheduler = \
            tf.keras.experimental.CosineDecay(self.args.lr, self.args.epochs)
            
    def get_config(self):
        return {
            'steps_per_epoch': self.steps_per_epoch,
            'init_lr': self.args.lr,}

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        step += self.initial_epoch * self.steps_per_epoch
        lr_epoch = (step / self.steps_per_epoch)
        return self.lr_scheduler(lr_epoch)


def callback_checkpoint(filepath, monitor, verbose, mode, save_best_only, save_weights_only):
    return ModelCheckpoint(
        filepath=filepath,
        monitor=monitor,
        verbose=verbose,
        mode=mode,
        save_best_only=save_best_only,
        save_weights_only=save_weights_only,
    )


def callback_epochlogger(filename, separator, append):
    return CSVLogger(filename=filename, separator=separator, append=append)


def callback_tensorboard(log_dir, batch_size):
    return TensorBoard(log_dir=log_dir, batch_size=batch_size)


def create_callbacks(args, path=None):
    path = path or os.path.join(args.result_path, args.dataset, args.stamp)
    
    callback_list = []
    if args.snapshot is None:
        if args.checkpoint or args.history or args.tensorboard:
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "model_desc.yml"), "w") as f:
                yaml.dump(vars(args), f, default_flow_style=False)

    if args.checkpoint:
        os.makedirs(os.path.join(path, 'checkpoint'), exist_ok=True)
        callback_list.append(
            callback_checkpoint(
                filepath=os.path.join(
                    path, 'checkpoint/{epoch:04d}_{val_acc:.4f}_{val_loss:.4f}.h5'),
                monitor="val_loss",
                verbose=1,
                mode="min",
                save_best_only=False,
                save_weights_only=True))

    if args.history:
        os.makedirs(os.path.join(path, 'history'), exist_ok=True)
        callback_list.append(
            callback_epochlogger(
                filename=os.path.join(path, 'history/epoch.csv'),
                separator=',',
                append=True))

    if args.tensorboard:
        os.makedirs(os.path.join(path, 'logs'), exist_ok=True)
        callback_list.append(
            callback_tensorboard(
                log_dir=os.path.join(path, 'logs'),
                batch_size=1))

    return callback_list