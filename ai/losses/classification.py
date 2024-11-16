import tensorflow as tf


def classification_loss(y_true, y_pred):
    return tf.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
