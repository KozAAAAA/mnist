import tensorflow as tf

import ai

train_ds, val_ds = ai.datasets.mnist()

for i in range(100):
    for images, labels in train_ds:
        images = images[..., tf.newaxis]

    for images, labels in val_ds:
        images = images[..., tf.newaxis]
