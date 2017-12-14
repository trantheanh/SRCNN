#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 18:15:11 2017

@author: anh_tt
"""

from de.model.SRCNN import SRCNN
import tensorflow as tf

with tf.Session() as sess:
    srcnn = SRCNN(sess,
                  image_size = 852,
                  label_size = 852,
                  image_channel = 3,
                  batch_size = 1,
                  num_epoch = 10,
                  checkpoint_dir = '/checkpoint',
                  sample_dir = '')
    
    srcnn.train()