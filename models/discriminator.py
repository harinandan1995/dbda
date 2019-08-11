import tensorflow as tf

from models.blocks import ConvBlock


class Discriminator(tf.keras.Model):

    def __init__(self, inp_image_channels, tar_image_channels, load_ckpt_path=None,
                 width=256, height=256, name='discriminator', **kwargs):

        super(Discriminator, self).__init__(name=name, **kwargs)

        self.width = width
        self.height = height
        self.inp_image_channels = inp_image_channels
        self.tar_image_channels = tar_image_channels

        self.model = self._generate_model()
        self.load_weights(load_ckpt_path)

    def call(self, inputs, training=None, mask=None):

        self.model(inputs, training=training)

    def _generate_model(self):

        inp = tf.keras.layers.Input(shape=[self.height, self.width, self.inp_image_channels], name='input_image')
        tar = tf.keras.layers.Input(shape=[self.height, self.width, self.tar_image_channels], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])

        x = ConvBlock(64, 4, 2, False, True)(x)
        x = ConvBlock(128, 4, 2, False, True)(x)
        x = ConvBlock(256, 4, 2, False, True)(x)
        x = tf.keras.layers.ZeroPadding2D()(x)
        x = ConvBlock(512, 4, 1, True, False)(x)
        x = tf.keras.layers.ZeroPadding2D()(x)
        x = ConvBlock(1, 4, 1, False, False)(x)

        return tf.keras.Model(inputs=[inp, tar], outputs=x)

    def save_weights(self, filepath, overwrite=True, save_format=None):
        self.model.save_weights(filepath, overwrite, save_format)

    def load_weights(self, filepath, by_name=False):

        if filepath is not None:
            self.model.load_weights(filepath)

