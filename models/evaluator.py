import numpy as np
import matplotlib.pyplot as plt

from models.generator import Generator


class FloorPlanGenerator:

    def __init__(self, dataset, gen_ckpt_path, out_dir, width=256, height=256):

        self.dataset = dataset
        self.width = width
        self.height = height
        self.out_dir = out_dir

        self.generator = Generator(1, 13, gen_ckpt_path, width, height)

    def evaluate(self, max_samples=10):

        dataset = self.dataset.batch(1)

        for index, data in dataset.enumerate():

            if index >= max_samples > 0:
                return

            wa, d, wi = data['wall_mask'], data['door_mask'], data['window_mask']
            r, s = data['room_mask'], data['shape_mask']

            fig = plt.figure(figsize=(8, 8))
            fig.add_subplot(4, 4, 1)
            wall_mask = np.rollaxis(wa[0].numpy(), 2, 0)[0]
            plt.imshow(wall_mask)
            fig.add_subplot(4, 4, 2)
            door_mask = np.rollaxis(d[0].numpy(), 2, 0)[0]
            plt.imshow(door_mask)
            fig.add_subplot(4, 4, 3)
            window_mask = np.rollaxis(wi[0].numpy(), 2, 0)[0]
            plt.imshow(window_mask)

            for i in range(10):
                target_numpy = np.rollaxis(r[0].numpy(), 2, 0)
                fig.add_subplot(4, 4, i + 4)
                plt.imshow(target_numpy[i])

            fig.add_subplot(4, 4, 14)
            shape_mask = np.rollaxis(s[0].numpy(), 2, 0)[0]
            plt.imshow(shape_mask)

            gen_out = self.generator(data['shape_mask']).numpy()
            gen_out = np.rollaxis(gen_out[0], 2, 0)

            wall_mask = np.expand_dims(gen_out[0], axis=0)
            door_mask = np.expand_dims(gen_out[1], axis=0)
            window_mask = np.expand_dims(gen_out[2], axis=0)
            room_mask = gen_out[3:]

            fig = plt.figure(figsize=(8, 8))
            fig.add_subplot(2, 2, 1)
            plt.imshow(gen_out[0], cmap='hot', interpolation='nearest')
            fig.add_subplot(2, 2, 2)
            plt.imshow(gen_out[1], cmap='hot', interpolation='nearest')
            fig.add_subplot(2, 2, 3)
            plt.imshow(gen_out[2], cmap='hot', interpolation='nearest')
            fig.add_subplot(2, 2, 4)
            plt.imshow(self._mask_to_segmentation_image(room_mask))

            plt.show()

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
