# zip and map is useful to combine multiple dataset obj into a singl dataset obj
import tensorflow as tf


dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))  # dataset of random values


dataset1.element_spec  # gets the specification of the element of this dataset


dataset2 = tf.data.Dataset.from_tensor_slices(
    (tf.random.uniform([4]),
    tf.random.uniform([4, 100], maxval = 100, dtype = tf.int32))
)

dataset2.element_spec


dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

dataset3.element_spec


# mapping
dataset = tf.data.Dataset.range(100)

# loops through dataset and convert to list
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))









































































































