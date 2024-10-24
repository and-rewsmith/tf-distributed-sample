import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Define the multi-worker strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Function to create the dataset


def make_dataset():
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(10000)
    return dataset

# Function to shard the dataset based on the worker index


def dataset_fn(input_context):
    global_batch_size = 64
    per_replica_batch_size = input_context.get_per_replica_batch_size(global_batch_size)

    # Create the base dataset without batching
    dataset = make_dataset()

    # Shard the dataset across workers
    dataset = dataset.shard(
        input_context.num_input_pipelines,
        input_context.input_pipeline_id
    )

    # Preprocess, batch, and prefetch
    dataset = dataset.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(per_replica_batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# Build a simple Keras model


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model


# Using the strategy scope
with strategy.scope():
    model = build_model()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    # Metrics to monitor loss and accuracy
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy'
    )

# Distribute the dataset across the workers
dataset = strategy.distribute_datasets_from_function(dataset_fn)

# Define the train step


@tf.function
def train_step(inputs):
    features, labels = inputs

    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

# Function to print shapes in a distributed context


def print_batch_shapes(features, labels):
    def _print_local_shapes(ctx):
        print(f"Replica {ctx.replica_id_in_sync_group} shapes - "
              f"Features: {features.shape}, Labels: {labels.shape}")
    strategy.run(_print_local_shapes, args=())


# Training loop
EPOCHS = 5

for epoch in range(EPOCHS):
    train_loss.reset_state()
    train_accuracy.reset_state()

    for batch_data in dataset:
        # Debug print for batch shapes using strategy.run
        features, labels = batch_data
        # strategy.run(lambda x, y: tf.print(
        #     "Per-replica shapes:",
        #     "features=", tf.shape(x),
        #     "labels=", tf.shape(y)
        # ), args=(features, labels))

        strategy.run(train_step, args=(batch_data,))

    print(
        f'Epoch {epoch+1}, '
        f'Loss: {train_loss.result():.4f}, '
        f'Accuracy: {train_accuracy.result() * 100:.2f}%'
    )
