import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import yaml
import argparse
import numpy as np
import pandas as pd

from common import set_seed
from common import get_arguments
from common import get_logger
from common import get_session
from callback import create_callbacks
from callback import OptionalLearningRateSchedule
from model.anynet import AnyNet
from dataloader import set_dataset
from dataloader import dataloader

import tensorflow as tf


def set_cfg(args, logger):
    path = os.path.join(args.result_path, args.dataset, args.model_name, str(args.stamp))
    initial_epoch = 0
    
    if os.path.isfile(os.path.join(path, 'history/epoch.csv')):
        df = pd.read_csv(os.path.join(path, 'history/epoch.csv'))
        if len(df) > 0:
            if len(df['epoch'].values) >= args.epochs:
                logger.info('{} Training already finished!!!'.format(args.stamp))
                return args, -1

            else:
                ckpt_list = sorted([d for d in os.listdir(os.path.join(path, 'checkpoint')) if 'h5' in d],
                                key=lambda x: int(x.split('_')[0]))
                print(ckpt_list)
                args.snapshot = os.path.join(path, 'checkpoint/{}'.format(ckpt_list[-1]))
                initial_epoch = int(ckpt_list[-1].split('_')[0])

    desc = yaml.full_load(open(os.path.join(path, 'model_desc.yml'), 'r'))
    for k, v in desc.items():
        if k in ['checkpoint', 'history', 'snapshot', 'gpus', 'src_path', 'data_path', 'result_path']:
            continue
        setattr(args, k, v)

    return args, initial_epoch


def create_model(args, logger):
    if 'anynet' in args.model_name.lower():
        model = AnyNet(args, name='anynet')
    elif 'regnet' in args.model_name.lower():
        pass
    else:
        raise ValueError()

    if args.snapshot:
        model.load_weights(args.snapshot)
        logger.info('Load model weights at {}'.format(args.snapshot))

    return model


def main():
    set_seed()
    args = get_arguments()
    assert args.model_name is not None, 'model_name must be set.'

    logger = get_logger("MyLogger")
    args, initial_epoch = set_cfg(args, logger)
    if initial_epoch == -1:
        # training was already finished!
        return

    get_session(args)
    for k, v in vars(args).items():
        logger.info("{} : {}".format(k, v))


    ##########################
    # Strategy
    ##########################
    # strategy = tf.distribute.MirroredStrategy()
    strategy = tf.distribute.experimental.CentralStorageStrategy()
    num_workers = strategy.num_replicas_in_sync
    assert args.batch_size % num_workers == 0

    logger.info('{} : {}'.format(strategy.__class__.__name__, num_workers))
    logger.info("GLOBAL BATCH SIZE : {}".format(args.batch_size))


    ##########################
    # Generator
    ##########################
    trainset, valset = set_dataset(args)
    train_generator = dataloader(args, trainset, 'train')
    val_generator = dataloader(args, valset, 'val', shuffle=False)
    
    steps_per_epoch = args.steps or len(trainset) // args.batch_size
    validation_steps = len(valset) // args.batch_size
    
    logger.info("TOTAL STEPS OF DATASET FOR TRAINING")
    logger.info("========== trainset ==========")
    logger.info("    --> {}".format(len(trainset)))
    logger.info("    --> {}".format(steps_per_epoch))

    logger.info("=========== valset ===========")
    logger.info("    --> {}".format(len(valset)))
    logger.info("    --> {}".format(validation_steps))


    ##########################
    # Model
    ##########################
    with strategy.scope():
        model = create_model(args, logger)
        if args.summary:
            from tensorflow.keras.utils import plot_model
            plot_model(model, to_file=os.path.join(args.src_path, 'model.png'), show_shapes=True)
            model.summary(line_length=130)
            return

        # optimizer
        scheduler = OptionalLearningRateSchedule(args, steps_per_epoch, initial_epoch)
        optimizer = tf.keras.optimizers.SGD(scheduler, momentum=.9, decay=.00005)

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=['acc']
        )


    ##########################
    # Callbacks
    ##########################
    callbacks = create_callbacks(
        args, 
        path=os.path.join(args.result_path, args.dataset, args.model_name, str(args.stamp)))
    logger.info("Build callbacks!")


    ##########################
    # Train
    ##########################
    model.fit(
        x=train_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        validation_data=val_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        initial_epoch=initial_epoch,
        verbose=1,
    )


if __name__ == "__main__":    
    main()