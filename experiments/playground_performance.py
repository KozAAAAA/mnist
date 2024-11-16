import time

import tensorflow as tf

import ai

ai.utils.allow_memory_growth()

images = tf.ones([16, 50, 50, 1])
classes = tf.random.uniform([16], 0, 10, tf.int32)

model1 = ai.models.image.ImageClassifier(10)
model1_optimizer = tf.optimizers.Adam(0.001)


@tf.function
def query(images, classes, training):
    model1_output = model1(images, training=training)
    model1_loss = ai.losses.classification_loss(classes, model1_output)
    model1_loss = tf.reduce_mean(model1_loss)
    return model1_loss


@tf.function
def train(images, classes):
    with tf.GradientTape() as tape:
        loss = query(images, classes, True)

    grads = tape.gradient(loss, model1.trainable_variables)
    model1_optimizer.apply_gradients(zip(grads, model1.trainable_variables))
    return loss


query(images, classes, False)
train(images, classes)

start = time.time()
for i in range(1000):
    query(images, classes, False)
end = time.time()

print(f'time: {(end - start) / 1000}')

start = time.time()
for i in range(1000):
    train(images, classes)
end = time.time()

print(f'time: {(end - start) / 1000}')

# time: 0.0009942286014556884
# time: 0.0037320854663848877

# time: 0.0003046741485595703
# time: 0.0028214848041534426
