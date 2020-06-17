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

        walls = data['wall_mask']
        doors = data['door_mask']
        windows = data['window_mask']
        corners = data['corner_mask']
        shape = data['shape_mask']

        model_input = tf.concat([walls, doors, windows], axis=3)
        model_input += tf.random.normal(model_input.shape, stddev=2)
        model_input = tf.cast(tf.greater(model_input, 0.7), tf.float32)

        output = self.model(model_input, training=True)

        loss = get_corner_loss(corners, output, shape, tf.constant(True), tf.constant(True))

        return loss, None

    def _epoch_end_call(self, mean_loss, current_epoch, total_epochs):

        if current_epoch % 5 == 0:
            self._save_weights(current_epoch)

        return {'prev epoch loss': mean_loss.numpy()}

    def _batch_end_call(self, loss, out, data, step, current_epoch, total_epochs) -> dict:

        return {'batch loss': loss.numpy()}
