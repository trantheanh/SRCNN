#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 18:04:07 2017

@author: anh_tt
"""
import sys
import os

# Server path
sys.path.insert(0, os.getcwd() + "/../../")

from de.model.SRCNN import SRCNN
import tensorflow as tf

# Reset default graph
tf.reset_default_graph()
with tf.Session() as sess:
    srcnn = SRCNN(sess,
                  image_size = 720,
                  label_size = 1440,
                  image_channel = 3,
                  batch_size = 1,
                  num_epoch = 9999,
                  checkpoint_dir = 'checkpoint',
                  sample_dir = '')
    
    srcnn.predict()