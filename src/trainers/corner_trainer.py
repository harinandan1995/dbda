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

        out = {
            'corners': output
        }

        return loss, out

    def _epoch_end_call(self, mean_loss, current_epoch, total_epochs):

        if current_epoch % 5 == 0:
            self._save_weights(current_epoch)

        return {'prev epoch loss': mean_loss.numpy()}

    def _batch_end_call(self, loss, out, data, step, current_epoch, total_epochs) -> dict:

        self._add_summaries(data, out, step, current_epoch)

        return {'batch loss': loss.numpy()}

    def _add_summaries(self, data, out, step, current_epoch):

        if step.numpy() % self.config.log.sum_step == 0:

            walls = data['wall_mask']
            doors = data['door_mask']
            windows = data['window_mask']
            wdw = tf.concat([walls, doors, windows], axis=3)

            corners = data['corner_mask']
            shape = data['shape_mask']
            pred_corners = out['corners']

            with self.train_sum_writer.as_default():
                
                tf.summary.image('{}/inp/shape'.format(current_epoch), shape,
                                 step.numpy() / self.config.log.sum_step, max_outputs=4)
                tf.summary.image('{}/inp/wdw'.format(current_epoch), wdw,
                                 step.numpy() / self.config.log.sum_step, max_outputs=4)
                tf.summary.image('{}/inp/corners'.format(current_epoch), tf.reduce_max(corners, axis=3, keepdims=True),
                                 step.numpy() / self.config.log.sum_step, max_outputs=4)
                tf.summary.image('{}/pred/corners'.format(current_epoch), tf.reduce_max(pred_corners, axis=3, keepdims=True),
                                 step.numpy() / self.config.log.sum_step, max_outputs=4)

            self.train_sum_writer.flush()
