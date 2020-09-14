import math

import tensorflow as tf

from src.models.losses import reconstruction_loss_v2
from src.trainers.trainer import ITrainer
from src.utils.config import ConfigParser
from src.nnutils.transforms import random_translate, random_rotate


class GLO(ITrainer):

    def __init__(self, config: ConfigParser.ConfigObject, model: tf.keras.Model,
                 train_dataset: tf.data.Dataset):

        super().__init__(config, model, train_dataset)
        self.lat_optim = self.optimizer[0]
        self.gen_optim = self.optimizer[1]
        self.lat_epoch_loss = tf.metrics.Mean()
        self.lat_var = tf.Variable(tf.random.normal([self.config.batch_size, self.config.lat.dim]),
                                   trainable=True, validate_shape=True)

    def _batch_start_call(self, data, step, current_epoch, total_epochs):

        walls = data['wall_mask']
        self.lat_var.assign(tf.random.normal([walls.shape[0], self.config.lat.dim]))

    @tf.function
    def _train_step(self, step, data):

        walls = data['wall_mask']
        doors = data['door_mask']
        windows = data['window_mask']
        shape = data['shape_mask']
        rooms = data['room_mask']
        room_type = data['room_types']

        dc = tf.expand_dims(data['door_count'], 1)
        wic = tf.expand_dims(data['window_count'], 1)
        cl = tf.expand_dims(data['cooling'], 1)
        ht = tf.expand_dims(data['heating'], 1)

        meta_input = tf.concat([room_type, dc, wic, cl, ht], axis=1)
        wdw_target = tf.concat([walls, doors, windows], axis=3)

        lat_loss = self._train_latent_variable(
            shape, meta_input, wdw_target, rooms, room_type, self.config.lat.iter)

        gen_loss, wdw_gen_out, room_gen_out = self._train_generator(
            shape, meta_input, wdw_target, rooms, room_type, self.config.gen.iter)

        out = {
            'lat_loss': lat_loss,
            'wdw_gen_out': wdw_gen_out,
            'room_gen_out': room_gen_out,
            'wdw_target': wdw_target,
            'room_target': rooms
        }

        return gen_loss, out

    @tf.function
    def _train_latent_variable(self, shape, meta_input, wdw_target, room_target, room_type, iterations):

        latent_loss = tf.constant(0.0)
        for lat_step in tf.range(iterations):
            with tf.GradientTape() as latent_tape:
                wdw_gen_out, room_gen_out = self.model([shape, meta_input, self.lat_var], training=True)
                latent_loss = reconstruction_loss_v2(shape, wdw_gen_out, room_gen_out,
                                                     wdw_target, room_target, room_type)

            latent_gradients = latent_tape.gradient(latent_loss, self.lat_var)
            self.lat_optim.apply_gradients(zip([latent_gradients], [self.lat_var]))

            latent_norm = tf.norm(self.lat_var, axis=1, keepdims=True)
            latent_norm = tf.tile(latent_norm, [1, self.config.lat.dim])
            self.lat_var.assign(tf.divide(self.lat_var, latent_norm) * math.sqrt(self.config.lat.dim))

        return latent_loss

    @tf.function
    def _train_generator(self, shape, meta_input, wdw_target, room_target, room_type, iterations):

        generator_loss = tf.constant(0.0)
        wdw_gen_out = tf.zeros_like(wdw_target)
        room_gen_out = tf.zeros_like(room_target)

        for gen_step in tf.range(iterations):
            with tf.GradientTape() as gen_tape:
                wdw_gen_out, room_gen_out = self.model([shape, meta_input, self.lat_var], training=True)
                generator_loss = reconstruction_loss_v2(shape, wdw_gen_out, room_gen_out,
                                                        wdw_target, room_target, room_type)

            generator_gradients = gen_tape.gradient(generator_loss, self.model.trainable_weights)
            self.gen_optim.apply_gradients(zip(generator_gradients, self.model.trainable_weights))

        return generator_loss, wdw_gen_out, room_gen_out

    def _get_augmented_data(self, data):

        if self.config.rotate:
            data = random_rotate(data, self.config.batch_size)
        if self.config.trans.status:
            data = random_translate(data, self.config.trans.max_x,
                                self.config.trans.max_x, self.config.batch_size)

        return data

    def _get_optimizer(self):

        optim_list = []

        if self.config.lat.optim.type == 'adam':
            lr = self.config.lat.optim.lr
            beta1 = self.config.lat.optim.beta1

            optim_list.append(tf.keras.optimizers.Adam(lr, beta_1=beta1))

        if self.config.gen.optim.type == 'adam':
            lr = self.config.gen.optim.lr
            beta1 = self.config.gen.optim.beta1

            optim_list.append(tf.keras.optimizers.Adam(lr, beta_1=beta1))

        return optim_list

    def _batch_end_call(self, loss, out, data, step, current_epoch, total_epochs):

        self.lat_epoch_loss.update_state(out['lat_loss'])

        self._add_summaries(data, out, step, current_epoch)

        return {
            'lat_loss': out['lat_loss'].numpy(),
            'gen_loss': loss.numpy(),
        }

    def _add_summaries(self, data, out, step, current_epoch):

        if step.numpy() % self.config.log.sum_step == 0:

            with self.train_sum_writer.as_default():
                tf.summary.image('{}/inp/shape'.format(current_epoch), data["shape_mask"],
                                 step.numpy() / self.config.log.sum_step, max_outputs=4)
                tf.summary.image('{}/inp/wdw'.format(current_epoch), out["wdw_target"],
                                 step.numpy() / self.config.log.sum_step, max_outputs=4)
                tf.summary.image('{}/pred/wdw'.format(current_epoch), out["wdw_gen_out"],
                                 step.numpy() / self.config.log.sum_step, max_outputs=4)

            self.train_sum_writer.flush()

    def _epoch_end_call(self, mean_loss, current_epoch, total_epochs) -> dict:

        with self.train_sum_writer.as_default():
            tf.summary.scalar('loss/lat', self.lat_epoch_loss.result(), current_epoch.numpy(),
                              'Mean latent loss per epoch')

        out = {
            'lat_loss': self.lat_epoch_loss.result().numpy(),
            'gen_loss': mean_loss.numpy()
        }
        self.lat_epoch_loss.reset_states()

        if current_epoch % 5 == 0:
            self._save_weights(current_epoch)

        return out
