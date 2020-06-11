from src.models.corner_detector import CornerDetector
from src.models.losses import get_corner_loss
from src.utils.utils import *


class CornerDetectorTrainer:

    def __init__(self, dataset, width, height, save_summary=True, summary_dir='./summaries',
                 save_model_ckpt=True, ckpt_dir='./checkpoints'):

        self.dataset = dataset
        self.height = height
        self.width = width

        self.save_summary = save_summary
        self.summary_dir = os.path.join(summary_dir, get_day(), get_time())

        if self.save_summary:
            create_directory_if_not_exist(self.summary_dir)

        self.save_model_ckpt = save_model_ckpt
        self.ckpt_dir = os.path.join(ckpt_dir, get_day(), get_time())

        if self.save_model_ckpt or self.save_disc_ckpt:
            create_directory_if_not_exist(self.ckpt_dir)

        self.epoch_summary_writer = tf.summary.create_file_writer(self.summary_dir)

        tf.summary.record_if(self.save_summary)

        self.model_config = {
            'optimizer': 'adam',
            'lr': 2e-4
        }

    def train(self, epochs, batch_size, num_samples, shuffle=True,
              model_ckpt=None, model_config=None):

        if model_config is not None:
            self.model_config = {**self.model_config, **model_config}

        train_dataset = self._get_training_dataset(batch_size, num_samples, shuffle)

        model = CornerDetector(3, 17, model_ckpt, self.width, self.height)
        print(model.summary())

        optimizer = self._get_optimizers()
        loss_metric = tf.keras.metrics.Mean()

        for epoch in range(epochs):

            start = time.time()

            summary_writer = tf.summary.create_file_writer(os.path.join(self.summary_dir, str(epoch)))

            with summary_writer.as_default():

                for step, (wa, d, wi, e, r, c, s, rt, wc, dc, wic) in enumerate(train_dataset):

                    loss = self._train_step(model, wa, d, wi, c, s, optimizer)

                    tf.summary.scalar('loss', loss, step)

                    loss_metric(loss)

                    if step % 1 == 0:
                        print('Step %4f, Loss - %.7f' % (step, loss))

                    summary_writer.flush()

            with self.epoch_summary_writer.as_default():

                tf.summary.scalar('loss_epoch', loss_metric.result(), step=epoch,
                                  description='Average model loss per epoch')

                self.epoch_summary_writer.flush()

            print('After epoch %d, time: %d, Loss - %.7f' % (epoch+1, time.time() - start, loss_metric.result()))

            loss_metric.reset_states()

            self._save_checkpoints(epoch, model)

    @tf.function
    def _train_step(self, model, walls, doors, windows, corners, shape, optimizer):

        with tf.GradientTape() as grad_tape:

            model_input = tf.concat([walls, doors, windows], axis=3)
            model_input += tf.random.normal(model_input.shape, stddev=2)
            model_input = tf.cast(tf.greater(model_input, 0.7), tf.float32)

            output = model(model_input, training=True)

            loss = get_corner_loss(corners, output, shape, tf.constant(True), tf.constant(True))

        gradients = grad_tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss

    def _get_optimizers(self):

        optimizer = None

        if self.model_config['optimizer'] == 'adam':

            lr = self.model_config['lr']
            optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)

        return optimizer

    def _get_training_dataset(self, batch_size, num_samples, shuffle):

        train_dataset = self.dataset

        if shuffle:
            train_dataset = train_dataset.shuffle(100)

        if num_samples > batch_size:
            train_dataset = train_dataset.take(num_samples)

        train_dataset = train_dataset.batch(batch_size)

        return train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def _save_checkpoints(self, epoch, model):

        if self.save_model_ckpt:
            ckpt_file_name = os.path.join(self.ckpt_dir, "model_%s_%d.h5" % (get_timestamp(), epoch))
            print('Saving model weights at %s' % ckpt_file_name)
            model.save_weights(ckpt_file_name)
