import tensorflow as tf


def discriminator_loss(real_out, gen_out):
    return - (tf.reduce_mean(real_out) - tf.reduce_mean(gen_out))


def generator_loss(gen_out):
    return - tf.reduce_mean(gen_out)
