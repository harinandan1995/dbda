import math

from models.generator import Generator
from models.losses import reconstruction_loss
from utils.utils import *


class GLOTrainer:

    def __init__(self, dataset, width, height, latent_dim,
                 save_summary=True, summary_dir='./summaries',
                 save_gen_ckpt=True, ckpt_dir='./checkpoints'):

        self.dataset = dataset
        self.height = height
        self.width = width
        self.latent_dim = latent_dim

        self.save_summary = save_summary
        self.summary_dir = os.path.join(summary_dir, get_day(), get_time())

        if self.save_summary:
            create_directory_if_not_exist(self.summary_dir)

        self.save_gen_ckpt = save_gen_ckpt
        self.ckpt_dir = os.path.join(ckpt_dir, get_day(), get_time())

        if self.save_gen_ckpt:
            create_directory_if_not_exist(self.ckpt_dir)

        self.epoch_summary_writer = tf.summary.create_file_writer(self.summary_dir)

        tf.summary.record_if(self.save_summary)

        self.gen_config = {
            'optimizer': 'adam',
            'lr': 2e-4
        }

        self.lat_config = {
            'optimizer': 'adam',
            'lr': 2e-3
        }

    def train(self, epochs, batch_size, num_samples, lat_iter, gen_iter, shuffle=True,
              load_gen_ckpt=None, gen_config=None, lat_config=None):

        if gen_config is not None:
            self.gen_config = {**self.gen_config, **gen_config}

        if lat_config is not None:
            self.lat_config = {**self.lat_config, **lat_config}

        train_dataset = self._get_training_dataset(batch_size, num_samples, shuffle)

        generator = Generator(1, [3, 10, 17], self.latent_dim, 12, load_gen_ckpt, self.width, self.height)
        print(generator.summary())

        generator_optimizer, latent_optimizer = self._get_optimizers()
        gen_loss_metric = tf.keras.metrics.Mean()
        lat_loss_metric = tf.keras.metrics.Mean()

        for epoch in range(epochs):

            start = time.time()

            summary_writer = tf.summary.create_file_writer(os.path.join(self.summary_dir, str(epoch)))

            with summary_writer.as_default():

                for step, (wa, d, wi, e, r, c, s, rt, wc, dc, wic) in enumerate(train_dataset):

                    gen_loss, lat_loss = self._train_step(
                        generator, wa, d, wi, r, c, s, rt, dc, wic,
                        generator_optimizer, latent_optimizer, lat_iter, gen_iter)

                    tf.summary.scalar('gen_loss', gen_loss, step)
                    tf.summary.scalar('lat_loss', lat_loss, step)

                    gen_loss_metric(gen_loss)
                    lat_loss_metric(lat_loss)

                    if step % 1 == 0:
                        print('Batch %4d, Loss - Gen %.7f Lat %.7f' % (step + 1, gen_loss, lat_loss))

            with self.epoch_summary_writer.as_default():

                tf.summary.scalar('gen_loss_epoch', gen_loss_metric.result(), step=epoch,
                                  description='Average generator loss per epoch')

                tf.summary.scalar('lat_loss_epoch', lat_loss_metric.result(), step=epoch,
                                  description='Average generator loss per epoch')

            print('After epoch %d, time: %d, Loss - gen: %.7f lat %.7f' % (epoch + 1, time.time() - start,
                                                                           gen_loss_metric.result(),
                                                                           lat_loss_metric.result()))

            gen_loss_metric.reset_states()
            lat_loss_metric.reset_states()

            self._save_checkpoints(epoch, generator)

    def _train_step(self, generator, walls, doors, windows, rooms, corners, shape,
                    room_type, door_count, window_count, generator_optimizer,
                    latent_optimizer, lat_iter, gen_iter):

        meta_input = tf.concat([room_type, door_count, window_count], axis=1)
        wdw_target = tf.concat([walls, doors, windows], axis=3)
        room_target = rooms
        corner_target = corners

        latent_variable = tf.Variable(tf.random.normal([meta_input.shape[0], self.latent_dim]),
                                      trainable=True, validate_shape=True)

        latent_loss, latent_variable = self._train_latent_variable(generator, latent_variable, latent_optimizer, shape,
                                                                   meta_input,
                                                                   wdw_target, room_target, corner_target, room_type,
                                                                   lat_iter)

        generator_loss = self._train_generator(generator, latent_variable, generator_optimizer, shape,
                                               meta_input, wdw_target, room_target, corner_target, room_type, gen_iter)

        return generator_loss, latent_loss

    def _train_latent_variable(self, generator, latent_variable, latent_optimizer, shape,
                               meta_input, wdw_target, room_target, corner_target, room_type, iterations):

        for lat_step in range(iterations):

            with tf.GradientTape() as latent_tape:

                wdw_gen_out, room_gen_out, corner_gen_out = generator([shape, meta_input, latent_variable],
                                                                      training=True)
                latent_loss = reconstruction_loss(shape, wdw_gen_out, room_gen_out, corner_gen_out,
                                                  wdw_target, room_target, corner_target, room_type)

            latent_gradients = latent_tape.gradient(latent_loss, latent_variable)

            latent_optimizer.apply_gradients(zip([latent_gradients], [latent_variable]))

            latent_norm = tf.norm(latent_variable, axis=1, keepdims=True)
            latent_norm = tf.tile(latent_norm, [1, self.latent_dim])
            latent_variable.assign(tf.divide(latent_variable, latent_norm) * math.sqrt(self.latent_dim))

            if lat_step % 5 == 0 or lat_step + 1 == iterations:
                print('Step %d Lat loss %f' % (lat_step + 1, latent_loss))

        return latent_loss, latent_variable

    def _train_generator(self, generator, latent_variable, generator_optimizer, shape,
                         meta_input, wdw_target, room_target, corner_target, room_type, iterations):

        for gen_step in range(iterations):

            with tf.GradientTape() as gen_tape:

                wdw_gen_out, room_gen_out, corner_gen_out = generator([shape, meta_input, latent_variable],
                                                                      training=True)
                generator_loss = reconstruction_loss(shape, wdw_gen_out, room_gen_out, corner_gen_out,
                                                     wdw_target, room_target, corner_target, room_type)

            generator_gradients = gen_tape.gradient(generator_loss, generator.trainable_weights)
            generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_weights))

            if gen_step % 5 == 0 or gen_step + 1 == iterations:
                print('Step %d Gen loss %f' % (gen_step + 1, generator_loss))

        return generator_loss

    def _get_optimizers(self):

        gen_optimizer = None

        if self.gen_config['optimizer'] == 'adam':
            lr = self.gen_config['lr']
            gen_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)

        latent_optimizer = None

        if self.lat_config['optimizer'] == 'adam':
            lr = self.lat_config['lr']
            latent_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)

        return gen_optimizer, latent_optimizer

    def _get_training_dataset(self, batch_size, num_samples, shuffle):

        train_dataset = self.dataset

        if shuffle:
            train_dataset = train_dataset.shuffle(100)

        if num_samples > batch_size:
            train_dataset = train_dataset.take(num_samples)

        train_dataset = train_dataset.batch(batch_size)

        return train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def _save_checkpoints(self, epoch, generator):

        if self.save_gen_ckpt:
            gen_ckpt_file_name = os.path.join(self.ckpt_dir, "gen_%s_%d.h5" % (get_timestamp(), epoch))
            print('Saving generator weights at %s' % gen_ckpt_file_name)
            generator.save_weights(gen_ckpt_file_name)
