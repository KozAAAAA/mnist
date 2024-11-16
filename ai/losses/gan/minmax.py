import tensorflow as tf

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_out, gen_out):
    real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_out), real_out, True)
    gen_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(gen_out), gen_out, True)
    return real_loss + gen_loss


def generator_loss(gen_out):
    return - discriminator_loss(tf.ones_like(gen_out), gen_out)
