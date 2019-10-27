import tensorflow as tf


gen_bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def convert_to_logits(y_pred):

    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    return tf.math.log(y_pred / (1 - y_pred))


def discriminator_loss(disc_real_output, disc_generated_output):

    real_loss = bce(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = bce(tf.zeros_like(disc_generated_output), disc_generated_output)

    return real_loss + generated_loss


def generator_loss(disc_generated_output, wdw_gen_out, room_gen_out, corner_gen_out,
                   wdw_target, room_target, corner_target, coeff):

    gan_loss = bce(tf.ones_like(disc_generated_output), disc_generated_output)

    # wdw_gen_logit = convert_to_logits(wdw_gen_out)
    # room_gen_logit = convert_to_logits(room_gen_out)
    # corner_gen_logit = convert_to_logits(corner_gen_out)

    # wdw_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(wdw_target, wdw_gen_logit, 1.6))
    # room_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(room_target, room_gen_logit, 0.6))
    # corner_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(corner_target, corner_gen_logit, 0.6))

    wdw_loss = gen_bce(wdw_target, wdw_gen_out)
    room_loss = gen_bce(room_target, room_gen_out)
    corner_loss = gen_bce(corner_target, corner_gen_out)

    total_gen_loss = 0.01 * gan_loss + coeff * (wdw_loss + room_loss + corner_loss)

    return total_gen_loss
