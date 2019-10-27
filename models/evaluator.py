import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from models.generator import Generator
from utils.utils import get_timestamp, create_directory_if_not_exist


class FloorPlanGenerator:

    """
    For a given set of inputs (shape mask etc.) generates masks for walls, windows, doors,
    rooms, corners using the generator from the checkpoint

    :param dataset: TFDataset from where the shape masks are stored
    :param gen_ckpt_path: Path to the checkpoint from where the generator weights are loaded
    :param
    """

    def __init__(self, dataset, gen_ckpt_path, out_dir='./out', width=128, height=128):

        self.dataset = dataset
        self.width = width
        self.height = height
        self.out_dir = out_dir

        create_directory_if_not_exist(self.out_dir)

        self.generator = Generator(1, [3, 10, 17], gen_ckpt_path, width, height)

    def evaluate(self, max_samples=10):

        dataset = self.dataset.batch(1)

        # Walls, doors, windows, rooms, corners, shape, room types, wall count,
        # door count, window count
        for index, (wa, d, wi, r, c, s, rt, wc, dc, wic) in dataset.enumerate():

            if index >= max_samples > 0:
                return

            self._plot_original_data(wa, d, wi, r, c, s, save_output=False)

            self._plot_generated_data(s, rt, wc, dc, wic, save_output=True)

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
            plt.savefig(os.path.join(self.out_dir,get_timestamp() + '_original.png'))

        plt.show()

    def _plot_generated_data(self, s, rt, wc, dc, wic, save_output=True):

        """
        Plots the heatmap of the generated data for the given shape of the building

        :param s: The shape mask for the generator
        :param save_output: True or False. When True a copy of the plot as an image is stored in
        the output directory
        """

        # wdw -> walls, doors, windows
        # Get the output of the generator
        count_input = tf.concat([rt, dc, wic], axis=1)
        wdw_gen_out, room_gen_out, corner_gen_out = self.generator([s, count_input])
        wdw_gen_out = np.rollaxis(wdw_gen_out.numpy()[0], 2, 0)
        room_gen_out = np.rollaxis(room_gen_out.numpy()[0], 2, 0)
        corner_gen_out = np.rollaxis(corner_gen_out.numpy()[0], 2, 0)

        fig = plt.figure(figsize=(10, 12))
        fig.add_subplot(5, 6, 1)
        plt.imshow(wdw_gen_out[0], cmap='hot', interpolation='nearest')
        fig.add_subplot(5, 6, 2)
        plt.imshow(wdw_gen_out[1], cmap='hot', interpolation='nearest')
        fig.add_subplot(5, 6, 3)
        plt.imshow(wdw_gen_out[2], cmap='hot', interpolation='nearest')

        for i in range(10):
            fig.add_subplot(5, 6, i + 4)
            plt.imshow(room_gen_out[i], cmap='hot', interpolation='nearest')

        for i in range(17):
            fig.add_subplot(5, 6, i + 14)
            plt.imshow(corner_gen_out[i], cmap='hot', interpolation='nearest')

        plt.savefig(os.path.join(self.out_dir,get_timestamp() + '_generated.png'))
        plt.show(block=True)

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
