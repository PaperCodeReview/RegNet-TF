import random
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


mean_std = {
    'cifar10' : [[0.491, 0.482, 0.447],
                 [0.247, 0.244, 0.262]],
    'cifar100': [[0.50707516, 0.48654887, 0.44091784], 
                 [0.20079844, 0.19834627, 0.20219835]],
    'imagenet': [[0.485, 0.456, 0.406],
                 [0.229, 0.224, 0.225]]
}

_EIG_VALS = np.array([[0.2175, 0.0188, 0.0045]])
_EIG_VECS = np.array(
    [[-0.5675, 0.7192, 0.4009], 
     [-0.5808, -0.0045, -0.8140], 
     [-0.5836, -0.6948, 0.4203]]
)

class Augment:
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode

        self.mean, self.std = mean_std[self.args.dataset]

    def __call__(self, x, shape):
        if self.args.dataset == 'imagenet':
            if self.mode == 'train':
                x = self._crop(x, shape)
                x = self._resize(x)
                x = self._random_hflip(x)
            else:
                x = self._resize(x, img_size=256)
                x = self._center_crop(x)
            
            x = tf.cast(x, tf.float32)
            x /= 255.
            if self.mode == 'train':
                x = self._lighting(x, 0.1, _EIG_VALS, _EIG_VECS)
            x -= self.mean
            x /= self.std

        elif 'cifar' in self.args.dataset:
            x = tf.cast(x, tf.float32)
            x /= 255.
            x -= self.mean
            x /= self.std
            if self.mode == 'train':
                x = self._random_hflip(x)
                x = self._crop(x, shape)

        else:
            raise ValueError()
        
        return x

    def _crop(self, x, shape, coord=[[[0., 0., 1., 1.]]]):
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            image_size=shape,
            bounding_boxes=coord,
            area_range=(.2, 1.),
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)

        offset_height, offset_width, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        x = tf.slice(x, [offset_height, offset_width, 0], [target_height, target_width, 3])
        return x

    def _center_crop(self, x):
        x = tf.image.central_crop(x, float(self.args.img_size)/256.)
        return x

    def _resize(self, x, img_size=None):
        img_size = img_size or self.args.img_size
        x = tf.image.resize(x, (img_size, img_size))
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _random_hflip(self, x):
        return tf.image.random_flip_left_right(x)

    def _lighting(self, x, alpha_std, eig_val, eig_vec):
        """Performs AlexNet-style PCA jitter."""
        if alpha_std == 0:
            return x
        alpha = np.random.normal(0, alpha_std, size=(1, 3))
        alpha = np.repeat(alpha, 3, axis=0)
        eig_val = np.repeat(eig_val, 3, axis=0)
        rgb = np.sum(eig_vec * alpha * eig_val, axis=1)
        rgb = rgb[::-1]
        x += rgb
        return x