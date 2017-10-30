#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-10-10 下午7:36
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

class Configure(object):
    """ global configuration """

    positive_data_dir = '/data/sunnymarkliu/tensorflow_text_classification/input/aclImdb/pos/'
    negative_data_dir = '/data/sunnymarkliu/tensorflow_text_classification/input/aclImdb/neg/'

    glove_word_embedding_file = '/data/sunnymarkliu/pretrained_models/glove/Wikipedia_2014/glove.6B.300d.txt'
    glove_word_embedding_matrix = '../input/glove_word_embedding_matrix.pkl'

    dataset_path = '../input/dataset.pkl'
