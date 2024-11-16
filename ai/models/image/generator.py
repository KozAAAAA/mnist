import tensorflow as tf


class ImageGenerator(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.noise_decoder = tf.keras.Sequential([
            # flat
            tf.keras.layers.Dense(7 * 7 * 256),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Reshape([7, 7, 256]),
            # conv without stride (7x7)
            tf.keras.layers.Conv2D(128, 5, 1, 'same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # t_conv with stride (14x14)
            tf.keras.layers.Conv2DTranspose(64, 5, 2, 'same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # conv without stride (14x14)
            tf.keras.layers.Conv2D(32, 5, 1, 'same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # t_conv with stride (28x28)
            tf.keras.layers.Conv2DTranspose(32, 5, 2, 'same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # conv without stride
            tf.keras.layers.Conv2D(1, 5, 1, 'same')
        ])

    def call(self, inputs, training=None, mask=None):
        return self.noise_decoder(inputs, training=training)
