import tensorflow as tf


def mnist(batch_size=16):
    train_ds, val_ds = tf.keras.datasets.mnist.load_data()

    train_ds = tf.data.Dataset.from_tensor_slices(train_ds) \
        .shuffle(1024) \
        .batch(batch_size) \
        .prefetch(8)

    val_ds = tf.data.Dataset.from_tensor_slices(val_ds) \
        .shuffle(1024) \
        .batch(batch_size) \
        .prefetch(8)

    return train_ds, val_ds
