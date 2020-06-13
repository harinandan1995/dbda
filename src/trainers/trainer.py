import os.path as osp

import tensorflow as tf
from tqdm import tqdm

from src.utils.config import ConfigParser
from src.utils.utils import get_date, get_time, create_dir_if_not_exists


class ITrainer:

    def __init__(self, config: ConfigParser.ConfigObject,
                 model: tf.keras.Model, train_dataset: tf.data.Dataset,
                 test_dataset: tf.data.Dataset = None):

        self.config = config

        self.model = model
        self._load_weights()

        self.train_dataset = self._prepare_dataset(train_dataset)
        self.test_dataset = self._prepare_dataset(test_dataset)

        self.optimizer = self._get_optimizer()
        self.total_epochs = tf.constant(config.epochs, dtype=tf.int64)

        self.checkpoint_dir, self.summary_dir = self._get_directories()
        self.train_sum_writer = tf.summary.create_file_writer(osp.join(self.summary_dir, 'train'))
        self.test_sum_writer = tf.summary.create_file_writer(osp.join(self.summary_dir, 'test'))

    def train(self, **kwargs):

        self.config.update(kwargs)

        epoch_loss = tf.metrics.Mean()
        epoch_bar = tqdm(tf.range(self.total_epochs, dtype=tf.int64),
                         position=0, unit='epoch')
        for epoch in epoch_bar:

            self._epoch_start_call(epoch, self.total_epochs)
            epoch_bar.set_description('Running {} epoch'.format(epoch))
            batch_bar = tqdm(position=1, leave=False)

            for step, data in self.train_dataset.enumerate():

                batch_bar.set_description('Training {} batch'.format(step))
                self._batch_start_call(data, step, epoch, self.total_epochs)

                if self.config.augment:
                    data = self._get_augmented_data(data)
                loss, out = self._train_step(step, data)

                epoch_loss.update_state(loss)
                self._update_bar(batch_bar, self._batch_end_call(
                    loss, out, data, step, epoch, self.total_epochs))

            batch_bar.close()

            with self.train_sum_writer.as_default():
                tf.summary.scalar('loss', epoch_loss.result(), epoch.numpy(), ' Mean loss per epoch')

            if self.config.run_test:
                self._run_test(epoch)

            epoch_bar.set_postfix(self._epoch_end_call(epoch_loss.result(), epoch, self.total_epochs))
            epoch_loss.reset_states()

    @tf.function
    def _train_step(self, step, data):

        with tf.GradientTape() as tape:
            loss, out = self._calculate_loss(data)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return loss, out

    @tf.function
    def _run_test(self, epoch):

        if self.test_dataset is None:
            raise AttributeError('Test dataset cannot be none if run_test value is True')

        loss_metric = tf.metrics.Mean()

        for step, data in self.test_dataset.enumerate():
            self._test_batch_start_call(data, step)
            loss, out = self._calculate_loss(data)
            loss_metric.update_state(loss)
            self._test_batch_end_call(loss, out, data, step, epoch)

        with self.test_sum_writer.as_default():
            tf.summary.scalar('loss', loss_metric.result(), epoch.numpy(), ' Mean loss per epoch')

    def _get_optimizer(self):

        if self.config.optim.type == 'adam':
            lr = self.config.optim.lr
            beta1 = self.config.optim.beta1

            return tf.keras.optimizers.Adam(lr, beta_1=beta1)

        return

    def _get_directories(self):

        sub_dir = osp.join(self.config.out_dir, get_date(), get_time())
        checkpoint_dir = osp.join(sub_dir, 'checkpoints')
        summary_dir = osp.join(sub_dir, 'summaries')

        create_dir_if_not_exists(checkpoint_dir, summary_dir)

        return checkpoint_dir, summary_dir

    def _load_weights(self):

        if self.config.ckpt_path != '' and osp.exists(self.config.ckpt_path):
            self.model.load_weights(self.config.ckpt_path)

    def _save_weights(self, epoch):

        file_path = osp.join(self.checkpoint_dir, "%s_%s_%d.h5" % (get_date(), get_time(), epoch.numpy()))
        self.model.save_weights(file_path)

    @staticmethod
    def _update_bar(bar: tqdm, logs: dict):

        bar.update(1)
        bar.set_postfix(logs)

    @tf.function
    def _calculate_loss(self, data):
        return NotImplementedError

    def _epoch_start_call(self, current_epoch, total_epochs):
        return

    def _epoch_end_call(self, mean_loss, current_epoch, total_epochs) -> dict:

        return {}

    def _batch_start_call(self, data, step, current_epoch, total_epochs):
        return

    def _batch_end_call(self, loss, out, data, step, current_epoch, total_epochs) -> dict:

        return {}

    def _test_batch_start_call(self, data, step):
        return

    def _test_batch_end_call(self, loss, out, data, step, current_epoch):
        return

    def _prepare_dataset(self, dataset):

        if dataset is None:
            return dataset

        if self.config.shuffle:
            dataset = dataset.shuffle(100)

        if self.config.num_samples > self.config.batch_size:
            dataset = dataset.take(self.config.num_samples)

        dataset = dataset.batch(self.config.batch_size, drop_remainder=True)

        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def _get_augmented_data(self, data):

        return
