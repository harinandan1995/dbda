import tensorflow as tf

from models.blocks import ConvBlock, ConvTransposeBlock


class CornerDetector(tf.keras.Model):

    def __init__(self, input_channels, output_channels,
                 load_ckpt_path=None, width=128, height=128,
                 name='corner_detector', **kwargs):

        super(CornerDetector, self).__init__(name=name, **kwargs)

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

        inputs = tf.keras.layers.Input(shape=[self.height, self.width, self.input_channels])

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        # Last layers
        corners = tf.keras.layers.Conv2DTranspose(self.output_channels, 4, strides=2, padding='same',
                                                  kernel_initializer=initializer, activation='sigmoid')(x)

        return tf.keras.Model(inputs=inputs, outputs=corners)

    @staticmethod
    def _get_downsample_stack():

        return [
            ConvBlock(16, 4, 2, False, False),
            ConvBlock(16, 4, 2, True, False),
            ConvBlock(32, 4, 2, True, False),
            ConvBlock(64, 4, 2, True, False),
            ConvBlock(64, 4, 2, True, False),
            ConvBlock(128, 4, 2, True, False),
            ConvBlock(128, 4, 2, True, False)
        ]

    @staticmethod
    def _get_upsample_stack():

        return [
            ConvTransposeBlock(128, 4, 2, True, False),
            ConvTransposeBlock(128, 4, 2, True, False),
            ConvTransposeBlock(64, 4, 2, True, False),
            ConvTransposeBlock(64, 4, 2, True, False),
            ConvTransposeBlock(32, 4, 2, True, False),
            ConvTransposeBlock(16, 4, 2, True, False),
            ConvTransposeBlock(16, 4, 2, True, False)
        ]

    def summary(self, line_length=None, positions=None, print_fn=None):

        return self.model.summary(line_length, positions, print_fn)

    def save_weights(self, filepath, overwrite=True, save_format=None):

        self.model.save_weights(filepath, overwrite, save_format)

    def load_weights(self, filepath, by_name=False):

        if filepath is not None:
            print('Loading model weights from %s' % filepath)
            self.model.load_weights(filepath)
