#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-12-13 下午9:57
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

from han_model import HierarchicalAttentionNetworks

# disable TF debug logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


print('---> build model')
# Built model and start training
# ==================================================
with tf.Graph().as_default(), tf.device('/gpu:1'):
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    session = tf.Session(config=session_conf)
    with session.as_default():
        han_model = HierarchicalAttentionNetworks(
            label_size=2,
            num_sentences=10,
            sequence_length=400,
            vocabulary_size=1000,
            embedding_dim=300,
            word_attention_size=100,
            sent_attention_size=100,
            word_encoder_bigru_num_units=100,
            embedding_trainable=False
        )