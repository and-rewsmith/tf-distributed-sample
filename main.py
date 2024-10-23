import tensorflow as tf

# Define the multi-worker strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Function to create and shard the dataset across workers


def make_dataset():
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(10000).batch(64)
    return dataset

# Function to shard the dataset based on the worker index


def dataset_fn(input_context):
    global_batch_size = 64
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)

    # Shard the dataset across workers
    dataset = make_dataset()
    dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    dataset = dataset.batch(batch_size)

    return dataset

# Build a simple Keras model


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model


# Using the strategy scope
with strategy.scope():
    model = build_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Distribute the dataset across the workers
    dataset = strategy.experimental_distribute_datasets_from_function(dataset_fn)

    log_dir = "./logs/fit/"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(dataset, epochs=5, callbacks=[tensorboard_callback])
