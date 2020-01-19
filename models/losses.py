import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

bce_without_logits = tf.keras.losses.BinaryCrossentropy(from_logits=False)
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

    wdw_loss = bce_without_logits(wdw_target, wdw_gen_out)
    room_loss = bce_without_logits(room_target, room_gen_out)
    corner_loss = bce_without_logits(corner_target, corner_gen_out)

    total_gen_loss = 0.1 * gan_loss + coeff * (wdw_loss + room_loss + corner_loss)

    return total_gen_loss


normal_dist = tfd.Normal(loc=0., scale=0.4)

@tf.function
def get_corner_loss(target, output, shape, add_shape_loss, add_weights):

    weights = tf.cond(add_weights, 
        lambda : normal_dist.prob(1 - tf.reduce_sum(target, 3, True)) * 20, 
        lambda : 1.0)

    loss = 20* bce_without_logits(target, output, sample_weight=weights)
    loss += tf.cond(add_shape_loss, 
        lambda : tf.reduce_mean((0.8 - shape) * output), lambda: 0.0)

    return loss


def reconstruction_loss(shape, wdw_gen_out, room_gen_out, corner_gen_out,
                        wdw_target, room_target, corner_target, room_type):

    weights = normal_dist.prob(1 - shape) * 5
    wdw_weights = normal_dist.prob(1 - tf.reduce_mean(wdw_target, 3, True)) * 10
    corner_weights = normal_dist.prob(1 - tf.reduce_sum(corner_target, 3, True)) * 10

    wdw_loss = bce_without_logits(wdw_target, wdw_gen_out, sample_weight=wdw_weights)
    room_loss = bce_without_logits(room_target, room_gen_out, sample_weight=weights)
    corner_loss = bce_without_logits(corner_target, corner_gen_out, sample_weight=corner_weights)

    shape_loss = tf.reduce_mean((0.8 - shape) * wdw_gen_out) + \
                 tf.reduce_mean((0.8 - shape) * room_gen_out) + \
                 tf.reduce_mean((0.8 - shape) * corner_gen_out)

    room_type_loss = tf.reduce_mean(tf.expand_dims(tf.expand_dims((0.5 - room_type), 1), 1) * room_gen_out)

    return 50 * wdw_loss + 10 * room_loss + 100 * corner_loss + 5 * room_type_loss + 5 * shape_loss


@tf.function
def reconstruction_loss_v2(shape, wdw_gen_out, room_gen_out,
                           wdw_target, room_target, room_type):

    weights = normal_dist.prob(1 - shape) * 5
    wdw_weights = normal_dist.prob(1 - tf.reduce_mean(wdw_target, 3, True)) * 10

    wdw_loss = bce_without_logits(wdw_target, wdw_gen_out, sample_weight=wdw_weights)
    room_loss = bce_without_logits(room_target, room_gen_out, sample_weight=weights)

    shape_loss = tf.reduce_mean((0.8 - shape) * wdw_gen_out) + \
                 tf.reduce_mean((0.8 - shape) * room_gen_out)

    room_type_loss = tf.reduce_mean(tf.expand_dims(tf.expand_dims((0.5 - room_type), 1), 1) * room_gen_out)

    return 50 * wdw_loss + 5 * room_loss + 5 * room_type_loss + 5 * shape_loss
