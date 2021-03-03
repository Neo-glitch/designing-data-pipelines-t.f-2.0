import numpy
import pandas as pd
import tensorflow as tf

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"


train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)  # downloads a file from specified url and stores as train.csv
test_file_path = tf.keras.utils.get_file("test.csv", TEST_DATA_URL)


LABEL_COLUMN = "survived"

# helper function to get the needed dataset for operation
def get_dataset(file_path, **kwargs):
    
    # prefetches a dataset that match param from file path of csv
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size = 5,
        label_name = LABEL_COLUMN,
        na_value = "?",
        num_epochs = 10,
        ignore_errors = True,
        **kwargs
    )
    return dataset


raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)


# helper to show the batch obtained from prefetched data
def show_batch(dataset):
    for batch, label in dataset.take(1): # iter trough 1st observatin in batch(i.e 1st batch only)
        for key, value in batch.items():
            print(f"{key}. {value.numpy()}")
            

show_batch(raw_test_data)


# csv already has col names, but if not best to do operation
CSV_COLUMNS = ["survived", "sex", "age", "n_siblings_spouses", "parch",
              "fare", "class", "deck", "embark_town", "alone"]

temp_dataset = get_dataset(train_file_path, column_names = CSV_COLUMNS)

show_batch(temp_dataset)


# selecting cols we want out of a csv
CSV_COLUMNS = ["survived", "class", "deck", "embark_town", "alone"]

temp_dataset = get_dataset(train_file_path, select_columns = CSV_COLUMNS)

show_batch(temp_dataset)




























































