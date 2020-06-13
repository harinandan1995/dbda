import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp


POSSIBLE_ANGLES = tf.constant([0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
DIST = tfp.distributions.Multinomial(total_count=1, probs=[0.25, 0.25, 0.25, 0.25])


def random_translate(data: dict, max_trans_x: int, max_trans_y: int, batch_size: int) -> dict:
    """
    Translates all masks in the data at random within the given limit.
    Translations along x and y are sampled at random within the limits
    :param data: A dictionary containing the masks. Only values for which the key contains
        mask in it are translated rest are left as is.
    :param max_trans_x: Maximum translation along x axis
    :param max_trans_y: Maximum translation along y axis
    :param batch_size: Batch size
    :return: A translated data
    """

    tran_x = tf.random.uniform([batch_size, 1], -max_trans_x, max_trans_x)
    tran_y = tf.random.uniform([batch_size, 1], -max_trans_y, max_trans_y)
    trans_xy = tf.concat([tran_x, tran_y], 1)

    tran_data = data.copy()
    for k, v in tran_data.items():
        if 'mask' in k:
            tran_data[k] = tfa.image.translate(v, trans_xy)

    return tran_data


@tf.function
def random_rotate(data, batch_size):
    """
    Rotates all the masks at random by one of the angles from the set (0, 90, 180, 270).
    Angles are sampled at random from the set.
    :param data: A dictionary containing the masks. Only values for which the keys contain
        mask in it are rotated, rest are left as is.
    :param batch_size: Batch size of the data
    :return: A rotated data
    """

    inds = tf.argmax(DIST.sample(batch_size), axis=1)
    angles = tf.gather(POSSIBLE_ANGLES, inds)

    tran_data = data.copy()
    for k, v in tran_data.items():
        if 'mask' in k:
            tran_data[k] = tfa.image.rotate(v, angles)

    return tran_data
