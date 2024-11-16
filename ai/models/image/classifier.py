import tensorflow as tf


class FlatImageClassifier(tf.keras.Model):

    def __init__(self, num_classes):
        super().__init__(name='flat_image_classifier')

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(256),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dense(num_classes)
        ])

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)


class ImageClassifier(tf.keras.Model):

    def __init__(self, num_classes):
        super().__init__(name='image_classifier')

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, 5, 2, 'same'),
            tf.keras.layers.Conv2D(16, 5, 2, 'same'),
            tf.keras.layers.Conv2D(32, 5, 2, 'same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes)
        ])

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)
