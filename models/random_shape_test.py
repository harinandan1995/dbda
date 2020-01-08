import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.ndimage.filters import maximum_filter

from models.generator import Generator
from models.losses import reconstruction_loss
from utils.utils import get_timestamp, create_directory_if_not_exist


class RandomShapeTester:
    """
    For a given set of inputs (shape mask etc.) generates masks for walls, windows, doors,
    rooms, corners using the generator from the checkpoint

    :param dataset: TFDataset from where the shape masks are stored
    :param gen_ckpt_path: Path to the checkpoint from where the generator weights are loaded
    :param
    """

    def __init__(self, dataset, latent_dim, gen_ckpt_path, out_dir='./out/random', width=128, height=128):

        self.dataset = dataset
        self.width = width
        self.height = height
        self.out_dir = out_dir
        self.latent_dim = latent_dim

        create_directory_if_not_exist(self.out_dir)

        self.latent_optimizer = tf.keras.optimizers.Adam(3e-2, beta_1=0.5)
        self.generator = Generator(1, [3, 10, 17], self.latent_dim, 12, gen_ckpt_path, width, height)

    def evaluate(self, max_samples=10):

        l_shape = np.zeros((self.width, self.height))
        # cv2.rectangle(l_shape, (32, 32), (96, 64), 1, -1)
        cv2.rectangle(l_shape, (42, 32), (84, 96), 1, -1)

        fig = plt.figure(figsize=(12, 12))
        fig.add_subplot(6, 6, 1)
        plt.imshow(l_shape)
        plt.savefig(os.path.join(self.out_dir, get_timestamp() + '_shape.png'))
        plt.show(block=True)

        l_shape = np.expand_dims(l_shape, 2)
        l_shape = np.expand_dims(l_shape, 0)

        dataset = self.dataset.batch(1)
        for index, (wa, d, wi, e, r, c, s, rt, wc, dc, wic) in dataset.enumerate():

            if index >= max_samples > 0:
                return

            meta_input = tf.concat([rt, dc, wic], axis=1)
            wdw_target = tf.concat([wa, d, wi], axis=3)
            room_target = r
            corner_target = c

            latent_variable = tf.Variable(tf.random.normal([meta_input.shape[0], self.latent_dim]),
                                          trainable=True, validate_shape=True)
            latent_variable = self._train_latent_variable(
                latent_variable, s, meta_input, wdw_target, room_target, corner_target, rt, 400)

            wdw_gen_out, room_gen_out, corner_gen_out = self.generator([l_shape, meta_input, latent_variable],
                                                                       training=False)

            wdw_gen_out = np.rollaxis(wdw_gen_out.numpy()[0], 2, 0)
            room_gen_out = np.rollaxis(room_gen_out.numpy()[0], 2, 0)
            corner_gen_out = np.rollaxis(corner_gen_out.numpy()[0], 2, 0)

            fig = plt.figure(figsize=(10, 12))
            fig.add_subplot(5, 6, 1)
            plt.imshow(self._filter_heatmap(wdw_gen_out[0]), cmap='hot', interpolation='nearest')
            fig.add_subplot(5, 6, 2)
            plt.imshow(self._filter_heatmap(wdw_gen_out[1]), cmap='hot', interpolation='nearest')
            fig.add_subplot(5, 6, 3)
            plt.imshow(self._filter_heatmap(wdw_gen_out[2]), cmap='hot', interpolation='nearest')

            for i in range(10):
                fig.add_subplot(5, 6, i + 4)
                plt.imshow(self._filter_heatmap(room_gen_out[i]), cmap='hot', interpolation='nearest')

            for i in range(17):
                fig.add_subplot(5, 6, i + 14)
                plt.imshow(self._filter_heatmap(corner_gen_out[i]), cmap='hot', interpolation='nearest')

            plt.savefig(os.path.join(self.out_dir, get_timestamp() + '_generated.png'))
            plt.show(block=True)

    def _filter_heatmap(self, heatmap, threshold=0.4):

        filtered_heatmap = np.zeros_like(heatmap)
        filtered_heatmap[heatmap > threshold] = 1

        return filtered_heatmap

    def _apply_nms(self, heatmap, window_size=10):

        return heatmap * (heatmap == maximum_filter(heatmap, footprint=np.ones((window_size, window_size))))

    def _train_latent_variable(self, latent_variable, shape, meta_input, wdw_target,
                               room_target, corner_target, room_type, iterations):

        for lat_step in range(iterations):

            with tf.GradientTape() as latent_tape:

                wdw_gen_out, room_gen_out, corner_gen_out = self.generator([shape, meta_input, latent_variable],
                                                                           training=False)
                latent_loss = reconstruction_loss(shape, wdw_gen_out, room_gen_out, corner_gen_out,
                                                  wdw_target, room_target, corner_target, room_type)

            latent_gradients = latent_tape.gradient(latent_loss, latent_variable)
            self.latent_optimizer.apply_gradients(zip([latent_gradients], [latent_variable]))

            latent_norm = tf.norm(latent_variable, axis=1, keepdims=True)
            latent_norm = tf.tile(latent_norm, [1, self.latent_dim])
            latent_variable.assign(tf.divide(latent_variable, latent_norm) * math.sqrt(self.latent_dim))

            if lat_step % 5 == 0 or lat_step + 1 == iterations:
                print('Step %d Lat loss %f' % (lat_step + 1, latent_loss))

        return latent_variable
