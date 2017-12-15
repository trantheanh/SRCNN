#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 18:15:11 2017

@author: anh_tt
"""
import sys
sys.path.insert(0, '/../')

from de.model.SRCNN import SRCNN
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 10, "Number of epoch [10]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 3840, "The size of image to train [3840]")

with tf.Session() as sess:
    srcnn = SRCNN(sess,
                  image_size = 3840,
                  label_size = 3840,
                  image_channel = 3,
                  batch_size = 4,
                  num_epoch = 10000,
                  checkpoint_dir = '/checkpoint',
                  sample_dir = '../../../Train')
    
    srcnn.train()