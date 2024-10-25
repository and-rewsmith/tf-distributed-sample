import json
import tensorflow as tf
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def mnist_dataset(batch_size):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = tf.cast(x_train, tf.float32) / 255.0
    y_train = tf.cast(y_train, tf.int64)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
    ).shuffle(60000).repeat().batch(batch_size)

    return train_dataset


def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28)),
        tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    # Explicitly use legacy optimizer
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.001)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    return model


per_worker_batch_size = 64
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])

strategy = tf.distribute.MultiWorkerMirroredStrategy()

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_dataset(global_batch_size)

with strategy.scope():
    multi_worker_model = build_and_compile_cnn_model()

# Remove the optimizer.build call since it's not needed with legacy optimizer

multi_worker_model.fit(
    multi_worker_dataset,
    epochs=3,
    steps_per_epoch=70
)
