import os
import sys
import yaml
import random
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime


def check_arguments(args):
        assert args.src_path is not None, 'src_path must be entered.'
        assert args.data_path is not None, 'data_path must be entered.'
        assert args.result_path is not None, 'result_path must be entered.'
        return args


def get_arguments():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--config",         type=str,       default='./config.yml')
    parser.add_argument("--num-model",      type=int,       default=100)
    parser.add_argument("--model_name",     type=str,       help='AnyNetXA')
    parser.add_argument("--stamp",          type=int,       default=0)
    parser.add_argument("--dataset",        type=str,       default='imagenet')

    # hyperparameter
    parser.add_argument("--batch_size",     type=int,       default=32, help="batch size per replica")
    parser.add_argument("--steps",          type=int,       default=0)
    parser.add_argument("--epochs",         type=int,       default=100)

    parser.add_argument("--optimizer",      type=str,       default='sgd')
    parser.add_argument("--lr",             type=float,     default=.001)
    parser.add_argument("--loss",           type=str,       default='crossentropy')

    parser.add_argument("--pad",            type=int,       default=0,              help='-1: square, 0: no, >1: set')

    # callback
    parser.add_argument("--checkpoint",     action='store_true')
    parser.add_argument("--history",        action='store_true')
    parser.add_argument("--evaluate",       action='store_true')
    parser.add_argument("--tensorboard",    action='store_true')

    # etc
    parser.add_argument('--src_path',       type=str,       default='.')
    parser.add_argument('--data_path',      type=str,       default=None)
    parser.add_argument('--result_path',    type=str,       default='./result')
    parser.add_argument('--snapshot',       type=str,       default=None)
    parser.add_argument("--gpus",           type=str,       default=-1)
    parser.add_argument("--summary",        action='store_true')
    parser.add_argument("--ignore_search",  type=str,       default='')

    return check_arguments(parser.parse_args())


def set_seed(SEED=42):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger.addHandler(screen_handler)
    return logger


def get_session(args):
    assert int(tf.__version__.split('.')[0]) >= 2.0
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.gpus != '-1':
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)


def search_same(args):
    search_ignore = ['checkpoint', 'history', 'snapshot', 'summary',
                     'src_path', 'data_path', 'result_path', 
                     'epochs', 'stamp', 'gpus', 'ignore_search']
    if len(args.ignore_search) > 0:
        search_ignore += args.ignore_search.split(',')

    initial_epoch = 0
    stamps = os.listdir(os.path.join(args.result_path, args.dataset))
    for stamp in stamps:
        try:
            desc = yaml.full_load(
                open(os.path.join(
                    args.result_path, '{}/{}/model_desc.yml'.format(args.dataset, stamp))))
        except:
            continue

        flag = True
        for k, v in vars(args).items():
            if k in search_ignore:
                continue
                
            if v != desc[k]:
                # if stamp == '200903_Thu_05_38_31':
                #     print(stamp, k, desc[k], v)
                flag = False
                break
        
        if flag:
            args.stamp = stamp
            try:
                df = pd.read_csv(
                    os.path.join(
                        args.result_path, 
                        '{}/{}/history/epoch.csv'.format(args.dataset, args.stamp)))
            except:
                continue

            if len(df) > 0:
                if int(df['epoch'].values[-1]+1) == args.epochs:
                    print('{} Training already finished!!!'.format(stamp))
                    return args, -1

                elif np.isnan(df['val_loss'].values[-1]) or np.isinf(df['val_loss'].values[-1]):
                    print('{} | Epoch {:04d}: Invalid loss, terminating training'.format(stamp, int(df['epoch'].values[-1]+1)))
                    return args, -1

                else:
                    ckpt_list = sorted(
                        [d for d in os.listdir(
                            os.path.join(
                                args.result_path, 
                                '{}/{}/checkpoint/query'.format(args.dataset, args.stamp))) if 'h5' in d],
                        key=lambda x: int(x.split('_')[0]))
                    
                    if len(ckpt_list) > 0:
                        args.snapshot = os.path.join(
                            args.result_path, 
                            '{}/{}/checkpoint/query/{}'.format(args.dataset, args.stamp, ckpt_list[-1]))
                        initial_epoch = int(ckpt_list[-1].split('_')[0])
                    else:
                        print('{} Training already finished!!!'.format(stamp))
                        return args, -1

            break
    
    return args, initial_epoch