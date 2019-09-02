import tensorflow as tf


class ConvBlock(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides,
                 apply_batch_norm=True, apply_dropout=False, padding='same'):

        super(ConvBlock, self).__init__()

        initializer = tf.random_normal_initializer(0., 0.02)

        self.apply_batch_norm = apply_batch_norm
        self.apply_dropout = apply_dropout

        self.conv_2d = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding=padding,
                                              kernel_initializer=initializer, use_bias=False)

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.activation = tf.keras.layers.LeakyReLU()

    def call(self, inputs, **kwargs):

        x = self.conv_2d(inputs)

        if self.apply_batch_norm:
            x = self.batch_norm(x)

        if self.apply_dropout:
            x = self.dropout(x)

        x = self.activation(x)

        return x


class ConvTransposeBlock(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides, apply_batch_norm=True, apply_dropout=False):

        super(ConvTransposeBlock, self).__init__()

        initializer = tf.random_normal_initializer(0., 0.02)

        self.apply_batch_norm = apply_batch_norm
        self.apply_dropout = apply_dropout

        self.conv_2d_trans = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding='same',
                                                             kernel_initializer=initializer, use_bias=False)

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.activation = tf.keras.layers.LeakyReLU()

    def call(self, inputs, **kwargs):

        x = self.conv_2d_trans(inputs)

        if self.apply_batch_norm:
            x = self.batch_norm(x)

        if self.apply_dropout:
            x = self.dropout(x)

        x = self.activation(x)

        return x