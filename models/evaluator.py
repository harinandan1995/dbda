import math
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.ndimage.filters import maximum_filter

from models.generator import Generator
from models.losses import reconstruction_loss
from utils.utils import get_timestamp, create_directory_if_not_exist


class FloorPlanGenerator:
    """
    For a given set of inputs (shape mask etc.) generates masks for walls, windows, doors,
    rooms, corners using the generator from the checkpoint

    :param dataset: TFDataset from where the shape masks are stored
    :param gen_ckpt_path: Path to the checkpoint from where the generator weights are loaded
    :param
    """

    def __init__(self, dataset, latent_dim, gen_ckpt_path, out_dir='./out', width=128, height=128):

        self.dataset = dataset
        self.width = width
        self.height = height
        self.out_dir = out_dir
        self.latent_dim = latent_dim

        self.threshold = 2

        create_directory_if_not_exist(self.out_dir)

        self.latent_optimizer = tf.keras.optimizers.Adam(6e-2)
        self.generator = Generator(1, [3, 10, 17], self.latent_dim, 12, gen_ckpt_path, width, height)

    def evaluate(self, max_samples=10):

        dataset = self.dataset.batch(1)

        # Walls, doors, windows, rooms, corners, shape, room types, wall count,
        # door count, window count
        for index, (wa, d, wi, e, r, c, s, rt, wc, dc, wic) in dataset.enumerate():

            if index >= max_samples > 0:
                return

            self._plot_original_data(wa, d, wi, r, c, s, save_output=False)

            self._plot_generated_data(wa, d, wi, r, c, s, rt, wc, dc, wic, save_output=False)

    def _plot_original_data(self, wa, d, wi, r, c, s, save_output=False):
        """
        Plots the masks of the original data

        :param wa: Wall mask
        :param d: Door mask
        :param wi: Window mask
        :param r: Room masks
        :param c: Corner masks
        :param s: Shape mask
        :param save_output: True or False. When True a copy of the plot as an image is stored in
        the output directory
        """

        fig = plt.figure(figsize=(12, 12))
        fig.add_subplot(6, 6, 1)
        wall_mask = np.rollaxis(wa[0].numpy(), 2, 0)[0]
        plt.imshow(wall_mask)
        fig.add_subplot(6, 6, 2)
        door_mask = np.rollaxis(d[0].numpy(), 2, 0)[0]
        plt.imshow(door_mask)
        fig.add_subplot(6, 6, 3)
        window_mask = np.rollaxis(wi[0].numpy(), 2, 0)[0]
        plt.imshow(window_mask)

        for i in range(10):
            target_numpy = np.rollaxis(r[0].numpy(), 2, 0)
            fig.add_subplot(6, 6, i + 4)
            plt.imshow(target_numpy[i])

        for i in range(17):
            target_numpy = np.rollaxis(c[0].numpy(), 2, 0)
            fig.add_subplot(6, 6, i + 14)
            plt.imshow(target_numpy[i])

        fig.add_subplot(6, 6, 31)
        shape_mask = np.rollaxis(s[0].numpy(), 2, 0)[0]
        plt.imshow(shape_mask)

        if save_output:
            plt.savefig(os.path.join(self.out_dir, get_timestamp() + '_original.png'))

        plt.show()

    def _plot_generated_data(self, wa, d, wi, r, c, s, rt, wc, dc, wic, save_output=True):

        """
        Plots the heatmap of the generated data for the given shape of the building

        :param s: The shape mask for the generator
        :param save_output: True or False. When True a copy of the plot as an image is stored in
        the output directory
        """

        # wdw -> walls, doors, windows
        # Get the output of the generator
        meta_input = tf.concat([rt, dc, wic], axis=1)
        wdw_target = tf.concat([wa, d, wi], axis=3)
        room_target = r
        corner_target = c

        latent_variable = tf.Variable(tf.random.normal([meta_input.shape[0], self.latent_dim]),
                                      trainable=True, validate_shape=True)

        wdw_gen_out, room_gen_out, corner_gen_out, latent_loss = self._train_latent_variable(
            latent_variable, s, meta_input, wdw_target, room_target, corner_target, rt, 300)

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
            plt.imshow(self._filter_heatmap(corner_gen_out[i]),
                       cmap='hot', interpolation='nearest')

        plt.savefig(os.path.join(self.out_dir, get_timestamp() + '_generated.png'))
        plt.show()

        if latent_loss < 3.5:
            self._generate_text_file(corner_gen_out, wdw_gen_out)

    def _generate_text_file(self, corners, walls):

        corner_points = []

        for i in range(13):
            filtered_corners = self._filter_heatmap(self._apply_nms(corners[i]))
            x, y = np.nonzero(filtered_corners)
            corner_points.append(zip(x, y))

        final_corner_pairs = []
        top_points = self._append_point(corner_points, [2, 5, 6, 9, 10, 11, 12])
        right_points = self._append_point(corner_points, [3, 6, 7, 8, 10, 11, 12])
        bottom_points = self._append_point(corner_points, [0, 4, 7, 8, 9, 11, 12])
        left_points = self._append_point(corner_points, [1, 4, 5, 8, 9, 10, 12])

        for i in range(13):

            for (x, y) in corner_points[i]:

                if i == 0:
                    # 2 5 6 9 10 11 12
                    final_corner_pairs.append(self._find_closest_top_point(x, y, top_points))
                elif i == 1:
                    # 3 6 7 8 10 11 12
                    final_corner_pairs.append(self._find_closest_top_point(x, y, right_points))
                elif i == 2:
                    # 0 4 7 8 9 11 12
                    final_corner_pairs.append(self._find_closest_top_point(x, y, bottom_points))
                elif i == 3:
                    # 1 4 5 8 9 10 12
                    final_corner_pairs.append(self._find_closest_top_point(x, y, left_points))
                elif i == 4:
                    final_corner_pairs.append(self._find_closest_top_point(x, y, top_points))
                    final_corner_pairs.append(self._find_closest_top_point(x, y, right_points))
                elif i == 5:
                    final_corner_pairs.append(self._find_closest_top_point(x, y, right_points))
                    final_corner_pairs.append(self._find_closest_top_point(x, y, bottom_points))
                elif i == 6:
                    final_corner_pairs.append(self._find_closest_top_point(x, y, left_points))
                    final_corner_pairs.append(self._find_closest_top_point(x, y, bottom_points))
                elif i == 7:
                    final_corner_pairs.append(self._find_closest_top_point(x, y, left_points))
                    final_corner_pairs.append(self._find_closest_top_point(x, y, top_points))
                elif i == 8:
                    final_corner_pairs.append(self._find_closest_top_point(x, y, left_points))
                    final_corner_pairs.append(self._find_closest_top_point(x, y, right_points))
                    final_corner_pairs.append(self._find_closest_top_point(x, y, top_points))
                elif i == 9:
                    final_corner_pairs.append(self._find_closest_top_point(x, y, right_points))
                    final_corner_pairs.append(self._find_closest_top_point(x, y, top_points))
                    final_corner_pairs.append(self._find_closest_top_point(x, y, bottom_points))
                elif i == 10:
                    final_corner_pairs.append(self._find_closest_top_point(x, y, left_points))
                    final_corner_pairs.append(self._find_closest_top_point(x, y, right_points))
                    final_corner_pairs.append(self._find_closest_top_point(x, y, bottom_points))
                elif i == 11:
                    final_corner_pairs.append(self._find_closest_top_point(x, y, left_points))
                    final_corner_pairs.append(self._find_closest_top_point(x, y, top_points))
                    final_corner_pairs.append(self._find_closest_top_point(x, y, bottom_points))
                elif i == 12:
                    final_corner_pairs.append(self._find_closest_top_point(x, y, left_points))
                    final_corner_pairs.append(self._find_closest_top_point(x, y, right_points))
                    final_corner_pairs.append(self._find_closest_top_point(x, y, top_points))
                    final_corner_pairs.append(self._find_closest_top_point(x, y, bottom_points))

        print(final_corner_pairs)

    @staticmethod
    def _append_point(points, indices):

        out_list = []

        for i in indices:
            out_list += points[i]

        return out_list

    def _find_closest_top_point(self, x, y, points):

        distance = 100000
        closest_point = None
        for (pt_x, pt_y) in points:
            if y + self.threshold >= pt_y >= y - self.threshold:
                if x - pt_x > 0 and abs(x - pt_x) < distance:
                    distance = abs(x - pt_x)
                    closest_point = (pt_x, (pt_y + y)/2)

        return closest_point

    def _find_closest_bottom_point(self, x, y, points):

        distance = 100000
        closest_point = None
        for (pt_x, pt_y) in points:
            if y + self.threshold >= pt_y >= y - self.threshold:
                if x - pt_x < 0 and abs(x - pt_x) < distance:
                    distance = abs(x - pt_x)
                    closest_point = (pt_x, (pt_y + y)/2)

        return closest_point

    def _find_closest_left_point(self, x, y, points):

        distance = 100000
        closest_point = None
        for (pt_x, pt_y) in points:
            if x + self.threshold >= pt_x >= x - self.threshold:
                if y - pt_y > 0 and abs(y - pt_y) < distance:
                    distance = abs(y - pt_y)
                    closest_point = ((pt_x + x)/2, pt_y)

        return closest_point

    def _find_closest_right_point(self, x, y, points):

        distance = 100000
        closest_point = None
        for (pt_x, pt_y) in points:
            if x + self.threshold >= pt_x >= x - self.threshold:
                if y - pt_y < 0 and abs(y - pt_y) < distance:
                    distance = abs(y - pt_y)
                    closest_point = ((pt_x + x)/2, pt_y)

        return closest_point

    def _train_latent_variable(self, latent_variable, shape, meta_input, wdw_target,
                               room_target, corner_target, room_type, iterations):

        latent_loss = 100

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

        return wdw_gen_out, room_gen_out, corner_gen_out, latent_loss

    def _filter_heatmap(self, heatmap, threshold=0.4):

        filtered_heatmap = heatmap
        filtered_heatmap[heatmap <= threshold] = 0

        return filtered_heatmap

    def _apply_nms(self, heatmap, window_size=5):

        return heatmap * (heatmap == maximum_filter(heatmap, footprint=np.ones((window_size, window_size))))

    @staticmethod
    def _mask_to_segmentation_image(mask):
        """
        Converts a mask image with dimensions HxWxC into a segmentation image with C labels

        :param mask: A HXWXC mask image
        :return: A segmentation image with C labels
        """
        num_colors = mask.shape[0]
        colors = np.random.randint(255, size=(num_colors, 3))

        segmentation_image = 255 * np.ones((mask.shape[1], mask.shape[2], 3), np.uint8)

        for i in range(mask.shape[0]):
            segmentation_image[mask[i] == 1] = colors[i]

        return segmentation_image
