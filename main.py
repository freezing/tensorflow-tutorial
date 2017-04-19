import matplotlib as matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
import pandas as pd
import numpy as np

import random

TRAIN_FILENAME = "./data/train.csv"
TEST_FILENAME = "./data/test.csv"
SUBMIT_FILENAME = "./submit/submit.csv"

TRAIN_PERCENTAGE = 1.0

EPOCHS = 10000
BATCH_SIZE = 200

# Learning rate setup
STARTING_LEARNING_RATE = 0.2
DECAY_STEPS = 500
DECAY_RATE = 0.96

MEAN_FACTOR = 0 # 255/2
SCALE_FACTOR = 1.0/255.0






# TensorFlow stuff

global_step = tf.Variable(0, trainable=False, name="global_step")
starting_learning_rate = tf.constant(STARTING_LEARNING_RATE)
learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step, DECAY_STEPS, DECAY_RATE,
                                           staircase=True, name="learning_rate_exponential_decay")

# Get input features with labels
train_X = tf.placeholder(tf.float32, [None, 28, 28, 1], name="input_features")
train_labels = tf.placeholder(tf.int32, [None], name="input_labels")

# Convert labels to one_hot vectors
train_Y = tf.one_hot(train_labels, 10, name="train_Y_one_hot")

is_training = tf.placeholder(tf.bool, [], name="is_training")


# Build Model
input_layer = (train_X - MEAN_FACTOR) * SCALE_FACTOR

# Convolutional Layer TODO: More RAM, can't use 64 and 32 fitlers in conv_1 and conv_2
conv_1 = tf.layers.conv2d(inputs=input_layer, filters=20, kernel_size=[5, 5], padding="same", name="conv_1")

pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=[2, 2], strides=2, name="max_pool_1")

# TODO: Add pooling
conv_2 = tf.layers.conv2d(inputs=pool_1, filters=20, kernel_size=[5, 5], padding="same", name="conv_2")
pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=[2, 2], strides=2, name="max_pool_2")



# Dense layer
# dense_1 = tf.layers.dense(inputs=input_layer,     units=200, activation=tf.nn.relu, name="hidden_1")
# dropout_1 = tf.layers.dropout(inputs=dense_1, rate=0.75, name="dropout_1")
#
# dense_2 = tf.layers.dense(inputs=dropout_1,       units=100, activation=tf.nn.relu, name="hidden_2")
# dropout_2 = tf.layers.dropout(inputs=dense_2, rate=0.75, name="dropout_2")
#
# dense_3 = tf.layers.dense(inputs=dropout_2,         units=60, activation=tf.nn.relu, name="hidden_3")
# dense_4 = tf.layers.dense(inputs=dense_3,         units=30, activation=tf.nn.relu, name="hidden_4")



# With CNN
flat_conv = tf.reshape(pool_2, [-1, 7 * 7 * 20])
# TODO: Buy more RAM for units=1024
dense_c = tf.layers.dense(inputs=flat_conv, units=1024, activation=tf.nn.relu, name="dense_c")
dropout_c = tf.layers.dropout(inputs=dense_c, rate=0.75, training=is_training, name="dropout_c")

# Output layer
logits = tf.layers.dense(inputs=dropout_c, units=10, activation=tf.nn.softmax, name="logits")


# Define loss function
loss = tf.losses.softmax_cross_entropy(onehot_labels=train_Y, logits=logits)

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

# Create single train_step from loss and optimizer
train_step = optimizer.minimize(loss, global_step=global_step)



# Metrics
is_correct = tf.equal(tf.argmax(logits, axis=1), tf.argmax(train_Y, axis=1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


# Model End




def batch_for_epoch(features, labels, epoch, batch_size):
    l = len(features)
    f = epoch * batch_size
    to = (epoch + 1) * batch_size

    if f >= l:
        f %= l
        to %= l
    elif to >= l:
        to = l

    epoch_indices = [x for x in range(f, to)]

    return features[epoch_indices], labels[epoch_indices]


def sample_batch(features, labels, batch_size):
    sampled_indices = np.random.choice(len(features), batch_size)
    return features[sampled_indices], labels[sampled_indices]


def read_and_prepare_data():
    train_data = pd.read_csv(TRAIN_FILENAME, header=0)

    permutation = np.random.permutation(len(train_data))

    train_labels = train_data["label"].values
    del train_data["label"]
    train_features = train_data.values

    train_features = train_features.reshape((len(train_features), 28, 28, 1))
    return train_features[permutation], train_labels[permutation]


def read_and_prepare_test_data():
    test_data = pd.read_csv(TEST_FILENAME, header=0)
    test_data = test_data.values.reshape((len(test_data), 28, 28, 1))
    return test_data


train_data_features, train_data_labels = read_and_prepare_data()

test_data_features = read_and_prepare_test_data()

# Split train and test data
train_indices = random.sample(range(len(train_data_features)), int(TRAIN_PERCENTAGE * len(train_data_features)))
dev_indices = []

search_index = set(train_indices)
for idx in range(len(train_data_features)):
    if idx not in search_index:
        dev_indices.append(idx)

dev_data_features, dev_data_labels = train_data_features[dev_indices], train_data_labels[dev_indices]
train_data_features, train_data_labels = train_data_features[train_indices], train_data_labels[train_indices]

# Main
train_losses_history = []

with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run(init)

    init_loss = session.run(loss, feed_dict={train_X: train_data_features, train_labels: train_data_labels,
                                             is_training: False})
    print("Initial loss: ", init_loss)

    for epochId in range(EPOCHS):
        # batch_train_X, batch_train_Y = sample_batch(train_data_features, train_data_labels, BATCH_SIZE)
        batch_train_X, batch_train_Y = batch_for_epoch(train_data_features, train_data_labels, epochId, BATCH_SIZE)

        _, train_loss = session.run([train_step, loss], feed_dict={train_X: batch_train_X, train_labels: batch_train_Y,
                                                                   is_training: True})
        train_losses_history.append(train_loss)

        if epochId % 10 == 0:
            # print("Loss[{:d}] = {:f}".format(epochId, train_loss))
            print("Learning rate: ", session.run(learning_rate))

            if epochId % 100 == 0:
                train_accuracy = session.run(accuracy, feed_dict={train_X: train_data_features,
                                                                  train_labels: train_data_labels,
                                                                  is_training: False})
                test_accuracy = session.run(accuracy, feed_dict={train_X: dev_data_features,
                                                                 train_labels: dev_data_labels,
                                                                 is_training: False})
                print("Train Accuracy[{:d}] = {:f}".format(epochId, train_accuracy))
                print("Dev Accuracy[{:d}] = {:f}".format(epochId, test_accuracy))
                print()


    # Start real data testing
    one_hot_predictions = session.run(logits, feed_dict={train_X: test_data_features, is_training: False})
    predictions = np.argmax(one_hot_predictions, axis=1)

    predictions = predictions.reshape((len(predictions), 1))
    image_ids = np.array([x + 1 for x in range(len(predictions))]).reshape((len(predictions), 1))

    image_id_with_prediction = np.concatenate((image_ids, predictions), axis=1)

    write_df = pd.DataFrame(data=image_id_with_prediction, columns=["ImageId", "Label"], dtype=np.int32)
    write_df.to_csv(SUBMIT_FILENAME, index=False)

# for t_loss in train_losses_history:
#     print(t_loss)

