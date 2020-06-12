import os.path as osp

import tensorflow as tf

from src.models.losses import get_corner_loss
from src.trainers.trainer import ITrainer
from src.utils.config import ConfigParser
from src.utils.utils import get_timestamp


class CornerDetectorTrainer(ITrainer):

    def __init__(self, config: ConfigParser.ConfigObject,
                 model: tf.keras.Model, train_dataset: tf.data.Dataset,
                 test_dataset: tf.data.Dataset = None):

        super(CornerDetectorTrainer, self).__init__(config, model, train_dataset, test_dataset)

    def _calculate_loss(self, data):

        walls, doors, windows, _, _, corners, shape, *_ = data

        model_input = tf.concat([walls, doors, windows], axis=3)
        model_input += tf.random.normal(model_input.shape, stddev=2)
        model_input = tf.cast(tf.greater(model_input, 0.7), tf.float32)

        output = self.model(model_input, training=True)

        loss = get_corner_loss(corners, output, shape, tf.constant(True), tf.constant(True))

        return loss, None

    def _epoch_end_call(self, mean_loss, current_epoch, total_epochs):

        ckpt_file_name = osp.join(self.checkpoint_dir, "model_%s_%d.h5" % (get_timestamp(), current_epoch))
        self.model.save_weights(ckpt_file_name)
