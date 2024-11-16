from argparse import ArgumentParser

import tensorflow as tf


class FlatImageClassifier(tf.keras.Model):

    def __init__(self, num_classes):
        super().__init__(name="flat_image_classifier")

        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256),
                tf.keras.layers.Dense(128),
                tf.keras.layers.Dense(num_classes),
            ]
        )

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)


class ImageClassifier(tf.keras.Model):

    def __init__(self, num_classes):
        super().__init__(name="image_classifier")

        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Reshape((28, 28, 1)),
                tf.keras.layers.Conv2D(8, 5, 2, "same"),
                tf.keras.layers.Conv2D(16, 5, 2, "same"),
                tf.keras.layers.Conv2D(32, 5, 2, "same"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(num_classes),
            ]
        )

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)


def main(args):

    train_ds, val_ds = tf.keras.datasets.mnist.load_data()

    train_ds = (
        tf.data.Dataset.from_tensor_slices(train_ds)
        .shuffle(1024)
        .batch(args.batch_size)
        .prefetch(8)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices(val_ds)
        .shuffle(1024)
        .batch(args.batch_size)
        .prefetch(8)
    )

    if args.model == "flat":
        model = FlatImageClassifier(10)
    elif args.model == "cnn":
        model = ImageClassifier(10)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=False,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    val_loss = tf.keras.metrics.Mean(name="val_loss")
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def val_step(images, labels):
        predictions = model(images, training=False)
        loss = loss_object(labels, predictions)

        val_loss(loss)
        val_accuracy(labels, predictions)

    for epoch in range(args.epochs):
        train_loss.reset_state()
        train_accuracy.reset_state()
        val_loss.reset_state()
        val_accuracy.reset_state()

        for images, labels in train_ds:
            train_step(images, labels)

        for images, labels in val_ds:
            val_step(images, labels)

        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {train_loss.result()}, "
            f"Accuracy: {train_accuracy.result()}, "
            f"Val Loss: {val_loss.result()}, "
            f"Val Accuracy: {val_accuracy.result()}"
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="flat", choices=["flat", "cnn"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--allow-memory-growth", action="store_true", default=False)
    args, _ = parser.parse_known_args()

    if args.allow_memory_growth:
        ai.utils.allow_memory_growth()

    main(args)
