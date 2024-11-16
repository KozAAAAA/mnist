from argparse import ArgumentParser

import tensorflow as tf
from tqdm import tqdm

import ai


def main(args):
    train_ds, val_ds = ai.datasets.mnist(512)

    discriminator = ai.models.image.ImageDiscriminator()
    generator = ai.models.image.ImageGenerator()

    optimizer = tf.optimizers.Adam(0.0001)

    @tf.function
    def query(images, training):
        noise = tf.random.uniform([tf.shape(images)[0], 64], -1, 1)

        real_out = discriminator(images, training)
        gen_out = discriminator(generator(noise, training), training)

        d_loss = ai.losses.gan.minmax.discriminator_loss(real_out, gen_out)
        g_loss = ai.losses.gan.minmax.generator_loss(gen_out)

        return d_loss, g_loss

    @tf.function
    def train(images):
        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            d_loss, g_loss = query(images, True)

        d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
        g_grads = g_tape.gradient(g_loss, generator.trainable_variables)

        optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
        optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

        return d_loss, g_loss

    @tf.function
    def generate(num_images):
        noise = tf.random.uniform([num_images, 64], -1, 1)
        return tf.clip_by_value(generator(noise, False), -1, 1)

    md_loss = tf.metrics.Mean('discriminator')
    mg_loss = tf.metrics.Mean('generator')
    for i in range(1000):

        md_loss.reset_states()
        mg_loss.reset_states()
        with tqdm(total=60000) as pbar:
            for images, _ in train_ds:
                images = tf.cast(images, tf.float32)[..., tf.newaxis]
                images = (images - 127.5) / 127.5

                d_loss, g_loss = train(images)
                md_loss.update_state(d_loss)
                mg_loss.update_state(g_loss)
                pbar.update(tf.shape(images)[0].numpy())

        print('\n============================')
        print('Train epoch')
        print(f'Discriminator loss: {md_loss.result().numpy()}')
        print(f'Generator loss: {mg_loss.result().numpy()}')
        print('============================\n')

        md_loss.reset_states()
        mg_loss.reset_states()
        with tqdm(total=10000) as pbar:
            for images, _ in val_ds:
                images = tf.cast(images, tf.float32)[..., tf.newaxis]
                images = (images - 127.5) / 127.5

                d_loss, g_loss = query(images, False)
                md_loss.update_state(d_loss)
                mg_loss.update_state(g_loss)
                pbar.update(tf.shape(images)[0].numpy())

        print('\n============================')
        print('Validation epoch')
        print(f'Discriminator loss: {md_loss.result().numpy()}')
        print(f'Generator loss: {mg_loss.result().numpy()}')
        print('============================\n')

        ai.utils.show_images(generate(10))


if __name__ == '__main__':
    parser = ArgumentParser()
    # todo: pass arguments
    parser.add_argument('--allow-memory-growth', action='store_true', default=False)
    args, _ = parser.parse_known_args()

    if args.allow_memory_growth:
        ai.utils.allow_memory_growth()

    main(args)
