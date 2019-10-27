import tensorflow as tf

from models.blocks import ConvBlock, ConvTransposeBlock


class Generator(tf.keras.Model):

    def __init__(self, input_channels, output_channels, load_ckpt_path=None,
                 width=128, height=128, name='generator', **kwargs):

        super(Generator, self).__init__(name=name, **kwargs)

        self.width = width
        self.height = height
        self.output_channels = output_channels
        self.input_channels = input_channels

        self.model = self._generate_model()

        if load_ckpt_path is not None:
            self.load_weights(load_ckpt_path)

    def call(self, inputs, training=None, mask=None):

        return self.model(inputs, training=training)

    def _generate_model(self):

        down_stack = self._get_downsample_stack()
        up_stack = self._get_upsample_stack()

        initializer = tf.random_normal_initializer(0., 0.02)

        concat = tf.keras.layers.Concatenate()

        inputs = tf.keras.layers.Input(shape=[self.height, self.width, self.input_channels])

        count_inputs = tf.keras.layers.Input(shape=[12])

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        x = tf.keras.layers.Reshape((64,))(x)
        x = concat([x, count_inputs])

        x = tf.keras.layers.Dense(64, use_bias=False, input_shape=(76, ))(x)
        x = tf.keras.layers.Reshape((1, 1, 64))(x)

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = concat([x, skip])

        wdw = tf.keras.layers.Conv2DTranspose(self.output_channels[0], 4, strides=2, padding='same',
                                              kernel_initializer=initializer, activation='sigmoid')

        rooms = tf.keras.layers.Conv2DTranspose(self.output_channels[1], 4, strides=2, padding='same',
                                                kernel_initializer=initializer, activation='sigmoid')

        corners = tf.keras.layers.Conv2DTranspose(self.output_channels[2], 4, strides=2, padding='same',
                                                  kernel_initializer=initializer, activation='sigmoid')

        wdw_x = wdw(x)
        rooms_x = rooms(x)
        corners_x = corners(x)

        return tf.keras.Model(inputs=[inputs, count_inputs], outputs=[wdw_x, rooms_x, corners_x])

    @staticmethod
    def _get_downsample_stack():

        return [
            ConvBlock(16, 4, 2, False, True),
            ConvBlock(16, 4, 2, True, True),
            ConvBlock(32, 4, 2, True, True),
            ConvBlock(32, 4, 2, True, False),
            ConvBlock(32, 4, 2, True, False),
            ConvBlock(64, 4, 2, True, False),
            ConvBlock(64, 4, 2, True, False)
        ]

    @staticmethod
    def _get_upsample_stack():

        return [
            ConvTransposeBlock(128, 4, 2, True, True),
            ConvTransposeBlock(128, 4, 2, True, True),
            ConvTransposeBlock(128, 4, 2, True, True),
            ConvTransposeBlock(64, 4, 2, True, False),
            ConvTransposeBlock(64, 4, 2, True, False),
            ConvTransposeBlock(32, 4, 2, True, False)
        ]

    def summary(self, line_length=None, positions=None, print_fn=None):

        return self.model.summary(line_length, positions, print_fn)

    def save_weights(self, filepath, overwrite=True, save_format=None):

        self.model.save_weights(filepath, overwrite, save_format)

    def load_weights(self, filepath, by_name=False):

        if filepath is not None:
            print('Loading generator weights from %s' % filepath)
            self.model.load_weights(filepath)
