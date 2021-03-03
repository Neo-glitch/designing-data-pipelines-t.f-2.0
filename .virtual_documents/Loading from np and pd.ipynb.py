import numpy as np
import pandas as pd
import tensorflow as tf

DATA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

path = tf.keras.utils.get_file("mnist.npz", DATA_URL)


# loads arrays of the .npz pickle file as a dict(data)
with np.load(path) as data:
    train_examples = data["x_train"]
    train_labels = data["y_train"]
    test_examples = data["x_test"]
    test_labels = data["y_test"]


print(train_examples)


type(train_examples)


# conv numpy array to tf.data dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

type(train_dataset)


for values in train_dataset.take(1):  # iter through train_dataset and take just first value
    print(values)
    


csv_file = tf.keras.utils.get_file("heart.csv", "https://storage.googleapis.com/download.tensorflow.org/data/heart.csv")


df = pd.read_csv(csv_file)

df.sample(5)


# data type for each col
df.dtypes


# obj/string col "thal" to cat
df.thal = pd.Categorical(df["thal"])

df.thal = df.thal.cat.codes  # encodes this as number


target = df.pop("target")

dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

type(dataset)



















































