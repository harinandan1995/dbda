import tensorflow as tf
from tensorflow.keras import layers


class ConvBlock(layers.Layer):

    def __init__(self, filters, kernel_size, strides, apply_batch_norm=True, apply_dropout=False):

        super(ConvBlock, self).__init__()

        initializer = tf.random_normal_initializer(0., 0.02)

        self.apply_batch_norm = apply_batch_norm
        self.apply_dropout = apply_dropout

        self.conv_2d = layers.Conv2D(filters, kernel_size, strides, padding='same',
                                     kernel_initializer= initializer, use_bias=False)

        self.batch_norm = layers.BatchNormalization()
        self.dropout = layers.Dropout(0.5)
        self.activation = layers.LeakyReLU()

    def call(self, inputs):

        x = self.conv_2d(inputs)

        if self.apply_batch_norm:
            x = self.batch_norm(x)

        if self.apply_dropout:
            x = self.dropout(x)

        x = self.activation(x)


class ConvTransposeBlock(layers.Layer):

    def __init__(self, filters, kernel_size, strides, apply_batch_norm=True, apply_dropout=False):

        super(ConvTransposeBlock, self).__init__()

        initializer = tf.random_normal_initializer(0., 0.02)

        self.apply_batch_norm = apply_batch_norm
        self.apply_dropout = apply_dropout

        self.conv_2d_trans = layers.Conv2DTranspose(filters, kernel_size, strides, padding='same',
                                                    kernel_initializer=initializer, use_bias=False)

        self.batch_norm = layers.BatchNormalization()
        self.dropout = layers.Dropout(0.5)
        self.activation = layers.LeakyReLU()

    def call(self, inputs):

        x = self.conv_2d_trans(inputs)

        if self.apply_batch_norm:
            x = self.batch_norm(x)

        if self.apply_dropout:
            x = self.dropout(x)

        x = self.activation(x)