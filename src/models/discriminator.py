import tensorflow as tf

from src.models.blocks import ConvBlock


class Discriminator(tf.keras.Model):
    """
    Discriminator model which takes 3 inputs
    - Shape of the building (input)
    - Generated/real floor plan(walls, windows, doors, rooms and corners) for
      the shape of the building (target)
    - Value parameters such as window, door counts and room types

    :param inp_image_channels: Number of channels of the shape mask(input)
    :param tar_image_channels: List of number of channels for wall_door_window, rooms, corners
    :param load_ckpt_path: Path to the checkpoint if you want preloaded weights for the discriminator
    :param width: width of the input/target
    :param height: height of the input/target
    """

    def __init__(self, inp_image_channels, tar_image_channels, load_ckpt_path=None,
                 width=128, height=128, name='discriminator', **kwargs):

        super(Discriminator, self).__init__(name=name, **kwargs)

        self.width = width
        self.height = height
        self.inp_image_channels = inp_image_channels
        self.tar_image_channels = tar_image_channels

        self.model = self._generate_model()
        self.load_weights(load_ckpt_path)

    def call(self, inputs, training=None, mask=None):

        return self.model(inputs, training=training)

    def _generate_model(self):

        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[self.height, self.width, self.inp_image_channels], name='input_image')
        wdw_tar = tf.keras.layers.Input(shape=[self.height, self.width, self.tar_image_channels[0]], name='wdw_image')
        room_tar = tf.keras.layers.Input(shape=[self.height, self.width, self.tar_image_channels[1]], name='room_image')
        corner_tar = tf.keras.layers.Input(shape=[self.height, self.width, self.tar_image_channels[2]], name='corner_image')

        x = tf.keras.layers.concatenate([inp, wdw_tar, room_tar, corner_tar])

        x = ConvBlock(32, 4, 2, False, True)(x)
        x = ConvBlock(32, 4, 2, False, True)(x)
        x = ConvBlock(64, 4, 2, False, True)(x)
        x = tf.keras.layers.ZeroPadding2D()(x)
        x = ConvBlock(128, 4, 1, True, False, padding='valid')(x)
        x = tf.keras.layers.ZeroPadding2D()(x)
        x = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(x)

        return tf.keras.Model(inputs=[inp, wdw_tar, room_tar, corner_tar], outputs=x)

    def summary(self, line_length=None, positions=None, print_fn=None):

        return self.model.summary(line_length, positions, print_fn)

    def save_weights(self, filepath, overwrite=True, save_format=None):

        self.model.save_weights(filepath, overwrite, save_format)

    def load_weights(self, filepath, by_name=False):

        if filepath is not None:
            print('Loading discriminator weights from %s' % filepath)
            self.model.load_weights(filepath)
