#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 18:15:11 2017

@author: anh_tt
"""
import sys
# Server path
sys.path.insert(0, '/home/sepml/workspace/neural_artistic_style/dataGenerator/SRCNN-Tensorflow/de_lib/')

# Local path
#sys.path.insert(0, '/Users/anh_tt/Documents/workspace/MachineLearning/DE_LIB/DE_LIB')

from de.model.SRCNN import SRCNN
import tensorflow as tf
import de.data.image as dimage
import de.util.IO as io

#flags = tf.app.flags
#flags.DEFINE_integer("epoch", 10, "Number of epoch [10]")
#flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
#flags.DEFINE_integer("image_size", 3840, "The size of image to train [3840]")

data = io.read_dir("/../../../Train", ['jpg','jpeg','bmp','png','JPEG'])
print(data)

#with tf.Session() as sess:
#    srcnn = SRCNN(sess,
#                  image_size = 3840,
#                  label_size = 3840,
#                  image_channel = 3,
#                  batch_size = 1,
#                  num_epoch = 10000,
#                  checkpoint_dir = '/checkpoint',
#                  sample_dir = '../../../Train')
#    
#    srcnn.train()