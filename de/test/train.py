#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 18:15:11 2017

@author: anh_tt
"""
import sys
import os

# Server path
sys.path.insert(0, os.getcwd() + "/../../")

from de.model.SRCNN.SRCNN import SRCNN
import tensorflow as tf
import de.data.image as dimage
import de.util.IO as io

tf.reset_default_graph()
with tf.Session() as sess:
    srcnn = SRCNN(sess,
                  image_size = 810,
                  label_size = 1620,
                  image_channel = 3,
                  batch_size = 1,
                  num_epoch = 9999,
                  checkpoint_dir = 'checkpoint',
                  sample_dir = '/../../../Train') #/../../../Train
    
    srcnn.train()