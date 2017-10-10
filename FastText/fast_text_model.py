#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-10-10 下午3:35
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)


class FastText(object):
    def __init__(self):
        """
        init the model with hyper-parameters etc
        """
        pass

    def inference(self, x):
        """
        The forward calculation from x to y
        :return: 
        """
        pass

    def loss(self, batch_x, batch_y=None):
        """
        calc the loss
        :param batch_x: 
        :param batch_y: 
        :return: 
        """
        pass

    def optimize(self, batch_x, batch_y):
        """
        
        :param batch_x: 
        :param batch_y: 
        :return: 
        """
