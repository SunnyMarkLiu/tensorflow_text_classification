#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-10-16 上午11:18
"""
from __future__ import absolute_import, division, print_function

import math
import os
import sys
import time

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

from text_cnn_model import TextCNN
from utils import data_util
from conf.configure import Configure

# disable TF debug logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float('test_split_percentage', 0.2, 'Percentage of the training data to use for validation')
tf.flags.DEFINE_float('validate_split_percentage', 0.1, 'Percentage of the training data to use for validation')

# Model Hyperparameters
tf.flags.DEFINE_integer('embedding_dim', 300, 'Dimensionality of word embedding (default: 300)')
tf.flags.DEFINE_integer('max_document_length', 200, 'Max document length (default: 200)')
tf.flags.DEFINE_float('dropout_keep_ratio', 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float('filter_sizes', [2,3,4,5], "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer('num_filters', 128, 'Number of filters per filter size (default: 128)')

# Training parameters
tf.flags.DEFINE_integer("max_learning_rate", 0.01, "Max learning_rate when start training (default: 0.01)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("train_verbose_every_steps", 10, "Show the training info every steps (default: 100)")
tf.flags.DEFINE_integer("evaluate_every_steps", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every_steps", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("max_num_checkpoints_to_keep", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("decay_rate", 0.8, "Learning rate decay rate (default: 0.9)")
tf.flags.DEFINE_float("decay_steps", 2000, "Perform learning rate decay step (default: 10000)")
tf.flags.DEFINE_float("l2_reg_lambda", 1e-4, "L2 regulaization rate (default: 10000)")

timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
tf.flags.DEFINE_string("log_message", timestamp, "log dir message (default: timestamp)")

FLAGS = tf.flags.FLAGS

print('Training Parameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")

embedding_dimension = FLAGS.embedding_dim

# Data Preparation
# ==================================================
print('---> load train text dataset')
# datasets = data_util.get_datasets_20newsgroup()
# x_text, y = data_util.load_data_labels(datasets)
x_text, y = data_util.load_text_datasets()

print('---> build vocabulary according this text dataset')
document_len = np.array([len(x.split(" ")) for x in x_text])
print('document_length, max = {}, mean = {}, min = {}'.format(document_len.max(), document_len.mean(),
                                                              document_len.min()))
max_document_length = FLAGS.max_document_length
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=max_document_length)
# Maps documents to sequences of word ids in vocabulary
x = np.array(list(vocab_processor.fit_transform(x_text)))

vocabulary_size = len(vocab_processor.vocabulary_)
print('built vocabulary size: {:d}'.format(vocabulary_size))

print('---> Split train/validate/test set')
test_sample_index = -1 * int(FLAGS.test_split_percentage * float(len(y)))
x_train, x_test = x[:test_sample_index], x[test_sample_index:]
y_train, y_test = y[:test_sample_index], y[test_sample_index:]

valid_sample_index = -1 * int(FLAGS.validate_split_percentage * float(len(x_train)))
x_train, x_valid = x_train[:valid_sample_index], x_train[valid_sample_index:]
y_train, y_valid = y_train[:valid_sample_index], y_train[valid_sample_index:]

print("train/valid/test split: {:d}/{:d}/{:d}".format(len(y_train), len(y_valid), len(y_test)))
print('---> create train/valid data wapper')
train_data_wrapper = data_util.DataWrapper(x_train, y_train, istrain=True)
valid_data_wrapper = data_util.DataWrapper(x_valid, y_valid, istrain=True)
test_data_wrapper = data_util.DataWrapper(x_valid, y_valid, istrain=False)

print('---> build model')
# Built model and start training
# ==================================================
with tf.Graph().as_default(), tf.device('/gpu:2'):
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    session = tf.Session(config=session_conf)
    with session.as_default():
        text_cnn = TextCNN(embedding_dim=embedding_dimension,
                           sequence_length=max_document_length,
                           label_size=2,
                           vocabulary_size=vocabulary_size,
                           filter_sizes=FLAGS.filter_sizes,
                           num_filters=FLAGS.num_filters,
                           embedding_trainable=False,
                           l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define global training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=text_cnn.learning_rate)
        # train_op = optimizer.minimize(text_cnn.loss, global_step=global_step)
        grads_and_vars = optimizer.compute_gradients(
            text_cnn.loss)  # Compute gradients of `loss` for the variables in `var_list`.
        # some other operation to grads_and_vars, eg. cap the gradient
        # ...
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)  # Apply gradients to variables.

        # Keep track of gradient values and sparsity distribution (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram(name="{}/grad/hist".format(v.name), values=g)
                # The fraction of zeros in gradient
                grad_sparsity_summary = tf.summary.scalar("{}/grad/parsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(grad_sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "run", FLAGS.log_message))
        print("---> write logs and summaries to {}".format(out_dir))

        # Summaries for loss and accuracy
        learning_rate_summary = tf.summary.scalar("learning_rate", text_cnn.learning_rate)
        loss_summary = tf.summary.scalar("loss", text_cnn.loss)
        accuracy_summary = tf.summary.scalar("accuracy", text_cnn.accuracy)

        # Merge Summaries for train
        train_summary_op = tf.summary.merge(
            [learning_rate_summary, loss_summary, accuracy_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(logdir=train_summary_dir, graph=session.graph)

        # Merge Summaries for valid
        valid_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        valid_summary_dir = os.path.join(out_dir, "summaries", "valid")
        valid_summary_writer = tf.summary.FileWriter(logdir=valid_summary_dir, graph=session.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.join(out_dir, "checkpoints")
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_num_checkpoints_to_keep)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocabulary"))

        # Initialize all variables
        session.run(tf.global_variables_initializer())

        print('---> start training...')
        vocabulary = vocab_processor.vocabulary_
        print('---> load pre-trained word vectors')
        word_embeddings = data_util.load_glove_word_embeddings(vocabulary,
                                                               Configure.glove_word_embedding_file,
                                                               vector_size=300)
        print('---> assign model embedding matrix')
        session.run(tf.assign(ref=text_cnn.word_embedding_matrix, value=word_embeddings))


        def train_step(x_batch, y_batch, learning_rate_, dropout_keep_ratio):
            """ training step """
            feed_dict = {
                text_cnn.sentence: x_batch,
                text_cnn.labels: y_batch,
                text_cnn.learning_rate: learning_rate_,
                text_cnn.dropout_keep_ratio: dropout_keep_ratio
            }
            _, step, summaries, loss_, accuracy_ = session.run([train_op, global_step, train_summary_op,
                                                              text_cnn.loss, text_cnn.accuracy],
                                                             feed_dict=feed_dict)
            if step % FLAGS.train_verbose_every_steps == 0:
                time_str_ = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                print("train {}: step {}, loss {:g}, acc {:g}, learning_rate {:g}"
                      .format(time_str_, step, loss_, accuracy_, learning_rate_))

            train_summary_writer.add_summary(summaries, step)


        def valid_step(x_batch, y_batch, valid_writer=None):
            """ validate step """
            feed_dict = {
                text_cnn.sentence: x_batch,
                text_cnn.labels: y_batch,
                text_cnn.dropout_keep_ratio: 1.0
            }
            summaries, loss_, accuracy_ = session.run([valid_summary_op,
                                                     text_cnn.loss, text_cnn.accuracy],
                                                     feed_dict=feed_dict)
            if valid_writer:
                valid_writer.add_summary(summaries, current_step)
            return loss_, accuracy_


        max_learning_rate = FLAGS.max_learning_rate
        decay_rate = FLAGS.decay_rate
        decay_steps = FLAGS.decay_steps

        total_train_steps = FLAGS.epochs * (x_train.shape[0] // FLAGS.batch_size)

        print('---> total train steps: {}'.format(total_train_steps))
        counter = 0
        start_train_time = time.time()
        for epoch in range(FLAGS.epochs):
            print('-----> train epoch: {:d}/{:d}'.format(epoch + 1, FLAGS.epochs))
            for i in range(x_train.shape[0] // FLAGS.batch_size):
                learning_rate = max_learning_rate * math.pow(decay_rate, int(counter / decay_steps))
                counter += 1

                batch_x, batch_y = train_data_wrapper.next_batch(FLAGS.batch_size)

                train_step(batch_x, batch_y, learning_rate, FLAGS.dropout_keep_ratio)
                current_step = tf.train.global_step(session, global_step)

                if current_step % FLAGS.evaluate_every_steps == 0:
                    print('---> perform validate')

                    losses = []; accuracies = []
                    for _ in range(valid_data_wrapper.x.shape[0] // FLAGS.batch_size + 1):
                        val_x, val_y = valid_data_wrapper.next_batch(FLAGS.batch_size)
                        loss, accuracy = valid_step(val_x, val_y, valid_summary_writer)
                        losses.append(loss)
                        accuracies.append(accuracy)

                    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    print("valid {}: step: {:d}, loss {:g}, acc {:g}".format(time_str, current_step, np.mean(losses), np.mean(accuracies)))

                if current_step % FLAGS.checkpoint_every_steps == 0:
                    path = saver.save(session, checkpoint_prefix, global_step=global_step)
                    print('---> save model to checkpoint: {}'.format(path))

        end_train_time = time.time()
        print('============ end training, cast {:.2f}s ============'.format(end_train_time - start_train_time))
        print('---> predict test')

        def test_step(x_batch, y_batch):
            """ test step """
            feed_dict = {
                text_cnn.sentence: x_batch,
                text_cnn.labels: y_batch,
                text_cnn.dropout_keep_ratio: 1.0
            }
            accuracy_ = session.run(text_cnn.accuracy, feed_dict=feed_dict)
            return accuracy_


        accuracies = []
        for j in range(test_data_wrapper.x.shape[0] // FLAGS.batch_size + 1):
            test_x, test_y = valid_data_wrapper.next_batch(FLAGS.batch_size)
            accuracy = test_step(test_x, test_y)
            accuracies.append(accuracy * test_x.shape[0])

        accuracy = 1.0 * np.sum(accuracies) / test_data_wrapper.x.shape[0]
        print('test accuracy : {:g}'.format(accuracy))
