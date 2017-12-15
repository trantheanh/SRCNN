#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 11:28:15 2017

@author: anh_tt
"""

import tensorflow as tf
import os
import de.data.image as dimage
import scipy.ndimage
import time

class SRCNN(object):
    
    def __init__(self,
                 sess,
                 image_size = 30,
                 label_size = 30,
                 image_channel = 3,
                 batch_size = 8,
                 num_epoch = 1,
                 checkpoint_dir = '',
                 sample_dir = ''):
        self.sess = sess
        self.image_size = image_size
        self.label_size = label_size
        self.image_channel = image_channel
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.build_model()
        
    def build_model(self):
#        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.image_channel])
#        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.image_channel])
        
        self.images = tf.placeholder(tf.float32, [None, None, None, self.image_channel])
        self.labels = tf.placeholder(tf.float32, [None, None, None, self.image_channel])
        
        # Weight dictionary
        self.weights = {
                'w1': tf.Variable(tf.random_normal([9,9,self.image_channel,64],  stddev = 1e-3), name = 'w1'),
                'w2': tf.Variable(tf.random_normal([1,1,64,32], stddev = 1e-3), name = 'w2'),
                'w3': tf.Variable(tf.random_normal([5,5,32,self.image_channel],  stddev = 1e-3), name = 'w3')                
        }
        
        # Bias dictionary
        self.biases = {
                'b1': tf.Variable(tf.zeros([64]), name = 'b1'),
                'b2': tf.Variable(tf.zeros([32]), name = 'b2'),
                'b3': tf.Variable(tf.zeros([self.image_channel]), name = 'b3'),
        }
        
        conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides = [1,1,1,1], padding = 'SAME'))
        conv1 = conv1 + self.biases['b1']
        
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides = [1,1,1,1], padding = 'SAME'))
        conv2 = conv2 + self.biases['b2']
        
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, self.weights['w3'], strides = [1,1,1,1], padding = 'SAME'))
        conv3 = conv3 + self.biases['b3']
        
        # Prediction image
        self.pred = conv3
        
        # Loss MSE
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        
        # Saver
        self.saver = tf.train.Saver()
        
        return
    
    def train(self):
        path_cache = []
        
        # Optimizer
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        
        # Initializer
        self.initializer = tf.global_variables_initializer()
        
        self.sess.run(self.initializer)
        
        # Step counter
        counter = 0
        
        print("Start Training: ****")
        for epoch in range(self.num_epoch):
            path_cache = []
            start_time = time.time()
            while True:
                # Fill data to batch & cache the others
                batch, path_cache = dimage.load_batch(self.sample_dir, self.batch_size, path_cache)
                
                # Train
                batch_input, batch_label = self.get_data_from_batch(batch)
                counter += 1
                _, err = self.sess.run([self.optimizer, self.loss], feed_dict = {self.images: batch_input, self.labels : batch_label})
                
                # Log every 10 step
                if (counter % 10 == 0):
                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                          % ((epoch + 1), counter, time.time() - start_time, err))
                
                if (len(path_cache) == 0):
                    break
            end_time = time.time()
            print("Epoch " + str(epoch + 1) + " finish in " + str(end_time - start_time))
                
    def get_data_from_batch(self, batch):
        s = 8.
        scale = 1. * self.label_size / self.image_size
        batch_input = scipy.ndimage.interpolation.zoom(batch, (1.,1./(s*scale), 1./(s*scale), 1), prefilter = False)
        batch_input = scipy.ndimage.interpolation.zoom(batch_input, (1., scale/1., scale/1., 1), prefilter = False)
        batch_label = scipy.ndimage.interpolation.zoom(batch, (1., 1./s, 1./s, 1), prefilter = False)
        
        return batch_input, batch_label
    
    def save(self, step):
        print("**Saving checkpoint** ...")
        model_name = "SCRNN.model"
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)
        
        # Check directory is existed. If not -> create new
        if (not os.path.exists(checkpoint_dir)):
            os.makedirs(checkpoint_dir)
            
        # Save model
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step = step)
        
        return
    
    def load(self):
        print("**Loading checkpoint** ...")
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)
        
        # Load checkpoint
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        if (ckpt and ckpt.model_checkpoint_path):
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True

        return False
    
    