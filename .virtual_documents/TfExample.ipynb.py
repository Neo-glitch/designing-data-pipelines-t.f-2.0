import tensorflow as tf
import numpy as np
import pandas as pd


n_observations = int(10000)  # num of observations in to be created np datasets

# feature vector creation 4 features
feature0 = np.random.choice([False, True], n_observations)
feature1 = np.random.randint(0, 5, n_observations)
strings = np.array([b"a", b"b", b"c", b"d", b"e"])  # strings to match range of feature1
feature2 = strings[feature1]
feature3 = np.random.randn(n_observations)


features_dataset = tf.data.Dataset.from_tensor_slices(
    (feature0, feature1, feature2, feature3)
)


for f0, f1, f2, f3 in features_dataset.take(1):  # ret one row of feature tuple
    print(f0)
    print(f1)
    print(f2)
    print(f3)


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(feature0, feature1, feature2, feature3):
  """
  Creates a tf.train.Example message ready to be written to a file.
  the 4 features are the previously created 4 feature arrays
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
  feature = {
      'feature0': _int64_feature(feature0),
      'feature1': _int64_feature(feature1),
      'feature2': _bytes_feature(feature2),
      'feature3': _float_feature(feature3),
  }
  # Create a Features message using tf.train.Example.
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()  # ret tf.train.Example serialized as string


serialized_example = serialize_example(False, 4, b"c", 0.1234)
print(serialized_example)


# gets binary or serialized value in original tf.example
example_returned = tf.train.Example.FromString(serialized_example)

example_returned

































































































































