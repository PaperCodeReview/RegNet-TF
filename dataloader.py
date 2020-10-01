import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from augment import Augment

AUTO = tf.data.experimental.AUTOTUNE

def set_dataset(args):
    if 'cifar' in  args.dataset:
        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

        b_trainset = unpickle(os.path.join(args.data_path, 'train'))
        b_valset = unpickle(os.path.join(args.data_path, 'test'))
        # split = .2
        trainset_temp = {}
        valset_temp = {}
        for i, l in enumerate(b_trainset[b'fine_labels']):
            if not l in trainset_temp.keys():
                temp = np.dstack((b_trainset[b'data'][i][:1024].reshape((32,32)),
                                b_trainset[b'data'][i][1024:2048].reshape((32,32)),
                                b_trainset[b'data'][i][2048:].reshape((32,32))))
                trainset_temp[l] = temp[np.newaxis,...]
            else:
                temp = np.dstack((b_trainset[b'data'][i][:1024].reshape((32,32)),
                                b_trainset[b'data'][i][1024:2048].reshape((32,32)),
                                b_trainset[b'data'][i][2048:].reshape((32,32))))
                trainset_temp[l] = np.append(trainset_temp[l], temp[np.newaxis,...], axis=0)
                
        for i, l in enumerate(b_valset[b'fine_labels']):
            if not l in valset_temp.keys():
                temp = np.dstack((b_valset[b'data'][i][:1024].reshape((32,32)),
                                b_valset[b'data'][i][1024:2048].reshape((32,32)),
                                b_valset[b'data'][i][2048:].reshape((32,32))))
                valset_temp[l] = temp[np.newaxis,...]
            else:
                temp = np.dstack((b_valset[b'data'][i][:1024].reshape((32,32)),
                                b_valset[b'data'][i][1024:2048].reshape((32,32)),
                                b_valset[b'data'][i][2048:].reshape((32,32))))
                valset_temp[l] = np.append(valset_temp[l], temp[np.newaxis,...], axis=0)
                
        trainset = []
        valset = []
        for k, v in trainset_temp.items():
            for vv in v:
                trainset.append([vv, k])
                
        for k, v in valset_temp.items():
            for vv in v:
                valset.append([vv, k])
    else:
        trainset = pd.read_csv(
            os.path.join(
                args.data_path, '{}_trainset.csv'.format(args.dataset)
            )).values.tolist()
        valset = pd.read_csv(
            os.path.join(
                args.data_path, '{}_valset.csv'.format(args.dataset)
            )).values.tolist()

    return np.array(trainset, dtype='object'), np.array(valset, dtype='object')

#############################################################################
def dataloader(args, datalist, mode, shuffle=True):
    """
    Dataloader for cross-entropy loss
    
    Argument
        args : Namespace of arguments
        datalist : trainset or valset
        mode : train or val
        shuffle : shuffle or not
    
    Return
        tf.data pipeline
    """
    def fetch_dataset(path, y):
        x = tf.io.read_file(path)
        return tf.data.Dataset.from_tensors((x, y))

    def augmentation(img, label, shape):
        augment = Augment(args, mode)
        img = augment(img, shape)
        
        # one-hot encodding
        label = tf.one_hot(label, args.classes)
        return img, label

    def preprocess_image(img, label):
        if 'cifar' in args.dataset:
            shape = (32, 32, 3)
        else:
            shape = tf.image.extract_jpeg_shape(img)
            img = tf.io.decode_jpeg(img, channels=3)
        img, label = augmentation(img, label, shape)
        return (img, label)

    imglist, labellist = datalist[:,0].tolist(), datalist[:,1].tolist()
    if 'cifar' not in args.dataset:
        imglist = [os.path.join(args.data_path, i) for i in imglist]

    dataset = tf.data.Dataset.from_tensor_slices((imglist, labellist))
    dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(len(datalist))

    if 'cifar' not in args.dataset:
        dataset = dataset.interleave(fetch_dataset, num_parallel_calls=AUTO)
    dataset = dataset.map(preprocess_image, num_parallel_calls=AUTO)
    dataset = dataset.batch(args.batch_size)
    dataset = dataset.prefetch(AUTO)
    return dataset