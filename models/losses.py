import tensorflow as tf


bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generated_output):

    real_loss = bce(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = bce(tf.zeros_like(disc_generated_output), disc_generated_output)

    return real_loss + generated_loss


def generator_loss(disc_generated_output, gen_output, target, coeff):

    gan_loss = bce(tf.ones_like(disc_generated_output), disc_generated_output)

    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + coeff * l1_loss

    return total_gen_loss
