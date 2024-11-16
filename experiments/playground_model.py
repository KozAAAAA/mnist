import tensorflow as tf

import ai

ai.utils.allow_memory_growth()

images = tf.ones([16, 28, 28, 1])
images_flat = tf.ones([16, 28 * 28])
classes = tf.random.uniform([16], 0, 10, tf.int32)

model1 = ai.models.image.ImageClassifier(10)
model2 = ai.models.image.FlatImageClassifier(10)

model1_optimizer = tf.optimizers.Adam(0.001)
model2_optimizer = tf.optimizers.Adam(0.001)

with tf.GradientTape() as tape:
    model1_output = model1(images, training=True)
    model1_loss = ai.losses.classification_loss(classes, model1_output)
    model1_loss = tf.reduce_mean(model1_loss)
grads = tape.gradient(model1_loss, model1.trainable_variables)
model1_optimizer.apply_gradients(zip(grads, model1.trainable_variables))

with tf.GradientTape() as tape:
    model2_output = model2(images_flat, training=True)
    model2_loss = ai.losses.classification_loss(classes, model2_output)
    model2_loss = tf.reduce_mean(model2_loss)
grads = tape.gradient(model2_loss, model2.trainable_variables)
model2_optimizer.apply_gradients(zip(grads, model2.trainable_variables))

print('xD')
