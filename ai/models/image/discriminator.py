import tensorflow as tf


class ImageDiscriminator(tf.keras.Model):

    def __init__(self):
        super().__init__()

        # since discriminator is for classification it should be robust, thus, add
        # additional regularization like dropout to prevent from pixel attacks
        self.image_encoder = tf.keras.Sequential([
            # conv with stride (out = 14x14)
            tf.keras.layers.Conv2D(64, 5, 2, 'same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.3),
            # conv with steide (out = 7x7)
            tf.keras.layers.Conv2D(128, 3, 2, 'same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.3),
            # flatten + hidden layer
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.3),
            # prediction (LOGITS!)
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs, training=None, mask=None):
        return self.image_encoder(inputs, training=training)
