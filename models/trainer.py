from models.discriminator import Discriminator
from models.generator import Generator
from models.losses import discriminator_loss, generator_loss
from utils.utils import *


class GANTrainer:

    def __init__(self, dataset, width, height,
                 save_summary=True, summary_dir='./summaries',
                 save_gen_ckpt=True, save_disc_ckpt=True, ckpt_dir='./checkpoints'):

        self.dataset = dataset
        self.height = height
        self.width = width

        self.save_summary = save_summary
        self.summary_dir = os.path.join(summary_dir, get_day(), get_time())

        if self.save_summary:
            create_directory_if_not_exist(self.summary_dir)

        self.save_gen_ckpt = save_gen_ckpt
        self.save_disc_ckpt = save_disc_ckpt
        self.ckpt_dir = os.path.join(ckpt_dir, get_day(), get_time())

        if self.save_gen_ckpt or self.save_disc_ckpt:
            create_directory_if_not_exist(self.ckpt_dir)

        self.epoch_summary_writer = tf.summary.create_file_writer(self.summary_dir)

        tf.summary.record_if(self.save_summary)

        self.gen_config = {
            'optimizer': 'adam',
            'lr': 2e-4
        }

        self.disc_config = {
            'optimizer': 'adam',
            'lr': 2e-4
        }

    def train(self, epochs, batch_size, num_samples, shuffle=True, coeff=100,
              load_gen_ckpt=None, load_disc_path=None, gen_config=None, disc_config=None):

        if gen_config is not None:
            self.gen_config = {**self.gen_config, **gen_config}

        if disc_config is not None:
            self.disc_config = {**self.disc_config, **disc_config}

        train_dataset = self._get_training_dataset(batch_size, num_samples, shuffle)

        generator = Generator(1, [3, 10, 17], load_gen_ckpt, self.width, self.height)
        print(generator.summary())
        discriminator = Discriminator(1, [3, 10, 17], load_disc_path, self.width, self.height)
        print(discriminator.summary())

        generator_optimizer, discriminator_optimizer = self._get_optimizers()
        gen_loss_metric = tf.keras.metrics.Mean()
        disc_loss_metric = tf.keras.metrics.Mean()

        for epoch in range(epochs):

            start = time.time()

            summary_writer = tf.summary.create_file_writer(os.path.join(self.summary_dir, str(epoch)))

            with summary_writer.as_default():

                for step, (wa, d, wi, r, c, s, rt, wc, dc, wic) in enumerate(train_dataset):

                    gen_loss, disc_loss = self._train_step(
                        generator, discriminator, wa, d, wi, r, c, s, rt, dc, wic,
                        generator_optimizer, discriminator_optimizer, coeff)

                    tf.summary.scalar('gen_loss', gen_loss, step)
                    tf.summary.scalar('disc_loss', disc_loss, step)

                    gen_loss_metric(gen_loss)
                    disc_loss_metric(disc_loss)

                    if step % 1 == 0:
                        print('Step %4f, Loss - Gen %.7f Disc %.7f' % (step, gen_loss, disc_loss))

                    summary_writer.flush()

            with self.epoch_summary_writer.as_default():

                tf.summary.scalar('gen_loss_epoch', gen_loss_metric.result(), step=epoch,
                                  description='Average generator loss per epoch')

                tf.summary.scalar('dis_loss_epoch', disc_loss_metric.result(), step=epoch,
                                  description='Average generator loss per epoch')

                self.epoch_summary_writer.flush()

            print('After epoch %d, time: %d, Loss - gen: %.7f disc %.7f' % (epoch+1, time.time() - start,
                                                                            gen_loss_metric.result(),
                                                                            disc_loss_metric.result()))

            gen_loss_metric.reset_states()
            disc_loss_metric.reset_states()

            self._save_checkpoints(epoch, generator, discriminator)

    @staticmethod
    def _train_step(generator, discriminator, walls, doors, windows, rooms, corners, shape,
                     room_type, door_count, window_count, generator_optimizer, discriminator_optimizer, coeff):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            count_input = tf.concat([room_type, door_count, window_count], axis=1)
            wdw_gen_out, room_gen_out, corner_gen_out = generator([shape, count_input], training=True)

            wdw_target = tf.concat([walls, doors, windows], axis=3)
            room_target = rooms
            corner_target = corners

            disc_real_output = discriminator([shape, wdw_target, room_target, corner_target], training=True)
            disc_generated_output = discriminator([shape, wdw_gen_out, room_gen_out, corner_gen_out], training=True)

            gen_loss = generator_loss(disc_generated_output, wdw_gen_out, room_gen_out, corner_gen_out,
                                      wdw_target, room_target, corner_target, coeff)

            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    discriminator.trainable_variables))

        return gen_loss, disc_loss

    def _get_optimizers(self):

        gen_optimizer = None

        if self.gen_config['optimizer'] == 'adam':

            lr = self.gen_config['lr']
            gen_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)

        disc_optimizer = None

        if self.disc_config['optimizer'] == 'adam':
            lr = self.disc_config['lr']
            disc_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)

        return gen_optimizer, disc_optimizer

    def _get_training_dataset(self, batch_size, num_samples, shuffle):

        train_dataset = self.dataset

        if shuffle:
            train_dataset = train_dataset.shuffle(100)

        if num_samples > batch_size:
            train_dataset = train_dataset.take(num_samples)

        train_dataset = train_dataset.batch(batch_size)

        return train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def _save_checkpoints(self, epoch, generator, discriminator):

        if self.save_gen_ckpt:
            gen_ckpt_file_name = os.path.join(self.ckpt_dir, "gen_%s_%d.h5" % (get_timestamp(), epoch))
            print('Saving generator weights at %s' % gen_ckpt_file_name)
            generator.save_weights(gen_ckpt_file_name)

        if self.save_disc_ckpt:
            disc_ckpt_file_name = os.path.join(self.ckpt_dir, "disc_%s_%d.h5" % (get_timestamp(), epoch))
            print('Saving discriminator weights at %s' % disc_ckpt_file_name)
            discriminator.save_weights(disc_ckpt_file_name)
