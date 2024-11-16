import matplotlib.pyplot as plt
import tensorflow as tf


def show_images(images):
    if isinstance(images, tf.Tensor):
        images = images.numpy()

    plt.figure(figsize=(1, images.shape[0]))

    for i in range(images.shape[0]):
        plt.subplot(images.shape[0], 1, i + 1)
        plt.imshow(images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.show()
