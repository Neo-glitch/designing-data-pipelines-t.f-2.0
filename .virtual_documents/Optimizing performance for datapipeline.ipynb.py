import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import time


csv_file = tf.keras.utils.get_file("heart.csv", "https://storage.googleapis.com/download.tensorflow.org/data/heart.csv")

df = pd.read_csv(csv_file)
df.head()


df.thal = pd.Categorical(df.thal)
df.thal = df.thal.cat.codes


train_dataset = df.sample(frac = 0.8, random_state = 0)
test_dataset = df.drop(train_dataset.index)

train_labels = train_dataset.pop("target")  # does this in place 
test_labels = test_dataset.pop("target")


def norm(x, train_stats):
    # scale data, x = dataset to scale, train_stats is description of dataset
    return(x - train_stats["mean"]) / train_stats["std"]


train_stats = train_dataset.describe().T

normed_train_data = norm(train_dataset, train_stats)

normed_train_data.head()


test_stats = test_dataset.describe().T

normed_test_data = norm(test_dataset, test_stats)


model = keras.Sequential([
    # input shape is len of col of train data in 1d array form
    keras.layers.Dense(64, activation ="relu", input_shape = [len(normed_train_data.keys())]),
    keras.layers.Dense(64, activation = "relu"),
    keras.layers.Dense(1)
])

optimizer = keras.optimizers.RMSprop(0.001)
model.compile(loss = "mse", optimizer = optimizer,
             metrics = ["mse", "mae"])


history = model.fit(normed_train_data, train_labels, epochs = 10)


y_pred = model.predict(normed_test_data)


# y_pred = [x[0] for x in y_pred]

keras.losses.MAE(test_labels.values, y_pred)


model.evaluate(normed_test_data, test_labels)


model = keras.Sequential([
    # input shape is len of col of train data in 1d array form
    keras.layers.Dense(64, activation ="relu", input_shape = [len(normed_train_data.keys())]),
    keras.layers.Dense(64, activation = "relu"),
    keras.layers.Dense(1)
])

optimizer = keras.optimizers.RMSprop(0.001)
model.compile(loss = "mse", optimizer = optimizer,
             metrics = ["mse", "mae"])


# batching
dataset = tf.data.Dataset.from_tensor_slices(
    (normed_train_data.values, train_labels.values)
).batch(10)

history = model.fit(dataset, epochs = 100)


model = keras.Sequential([
    # input shape is len of col of train data in 1d array form
    keras.layers.Dense(64, activation ="relu", input_shape = [len(normed_train_data.keys())]),
    keras.layers.Dense(64, activation = "relu"),
    keras.layers.Dense(1)
])

optimizer = keras.optimizers.RMSprop(0.001)
model.compile(loss = "mse", optimizer = optimizer,
             metrics = ["mse", "mae"])


# prefetching
dataset = tf.data.Dataset.from_tensor_slices(
    (normed_train_data.values, train_labels.values)
).batch(10).prefetch(2)  # prefetches 2 batches at a time

history = model.fit(dataset, epochs = 100)


# mostly used when needed data is stored remotely
tf.data.Dataset.interleave(
    dataset,
    num_parallel_calls=tf.data.experimental.AUTOTUNE  # auto decide optimal num threads to use
)















































































































