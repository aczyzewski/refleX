import tensorflow as tf
import pandas as pd
import numpy as np

from vector_nn_utils import conv_net, model_fn


JOINT_VECTOR_FILE = '../metadata/joint_file.csv'
DATA_ORDER_FILE = '../metadata/data_order.csv'

# NUM_CLASSES = 2
# VECTOR_LENGTH = 240
# NUM_FEATURES = 7

FEATURE_NAME = 'Strong background'

joint_vector_df = pd.read_csv(JOINT_VECTOR_FILE, index_col=0)
data_order_df = pd.read_csv(DATA_ORDER_FILE, header=None, index_col=0, usecols=[0])
LAST_TRAIN_INDEX = 1782
# przy indeksowaniu od 0, od indeksu 1782 do końca zaczynają się indeksy testowe. (20% całego zbioru)

train_names_df = data_order_df.iloc[:LAST_TRAIN_INDEX, :]
test_names_df = data_order_df.iloc[LAST_TRAIN_INDEX:, :]

train_df = train_names_df.join(joint_vector_df, how='inner')
test_df = test_names_df.join(joint_vector_df, how='inner')
NUM_INPUT = train_df.shape[1]

train_x_df = train_df.iloc[:, :-7]
train_y_df = train_df.iloc[:, -7:]
test_x_df = test_df.iloc[:, :-7]
test_y_df = test_df.iloc[:, -7:]

train_x_np = train_x_df.values.astype(float)/255
test_x_np = test_x_df.values.astype(float)/255

train_y = train_y_df[FEATURE_NAME].values
test_y = test_y_df[FEATURE_NAME].values

# TODO toCaps
#learning_rate = 0.001
num_steps = 2000
batch_size = 128
dropout = 0.25

from vector_nn_utils import conv_net, model_fn

model = tf.estimator.Estimator(model_fn)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': train_x_np}, y=train_y,
    batch_size=batch_size, num_epochs=None, shuffle=True)

model.train(input_fn, steps=num_steps)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': test_x_np}, y=test_y,
    batch_size=batch_size, shuffle=False)

e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])