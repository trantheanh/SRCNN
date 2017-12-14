#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 13:20:54 2017

@author: anh_tt
"""

#from PIL import Image
import numpy as np
import scipy.ndimage
import scipy.misc
import de.util.IO as io

"""
Create on Wed Dec 13

@author: anh_tt

Load image base on path

Parameters:
    path - path to image file 

Return: 
    image: tensor 3D (width, height, channel)
"""
def load_image(path):
    image = scipy.misc.imread(path, flatten = False, mode = 'YCbCr')
    
    return image

"""
Create on Wed Dec 13

@author: anh_tt

Save image base on path

Parameters:
    image - numpy array corresponse image data
    path - path to image file 
    
Return: 
    image: image has just been saved
"""
def save_image(image, path):
    image = scipy.misc.imsave(path, image)
    
    return image

"""
Create on Wed Dec 13

@author: anh_tt

Scale image

Parameters:
    image - numpy array corresponse image data (width, height, color_channel)
    scale - scale image width and height to but not number of channel color
    
Return:
    image: new image after scale 
"""
def scale_image(image, scale):
    image = scipy.ndimage.interpolation.zoom(image, (scale, scale, 1), prefilter = False)
    
    return image

"""
Create on Wed Dec 13

@author: anh_tt

Normalize image to [0,1] instead of [0,255]

Parameters:
    image - numpy array corresponse image data (width, height, color_channel)
    
Return: 
    image - tensor 3D after each element divide 255
"""
def normalize_image(image):
    image = image / 255.
    return image

"""
Create on Wed Dec 13

@author: anh_tt

Crop image

Parameters:
    image - numpy array corresponse image data (width, height, color_channel)
    Crop - crop image to new width and height

Return:
    image: image after crop
"""
def crop_image(image, width, height):
    ## UNDER BUILDING
    cur_width, cur_height, _ = image.shape
    
    # Validate width & height
    if (cur_width < width):
        width = cur_width        
    
    if (cur_height < height):
        height - cur_height
    
    center = (cur_width/2, cur_height/2)
    return center

"""
Create on Wed Dec 13

@author: anh_tt

Load images from directory to batch and save them to h5. Name will be "batch_1", "batch_2"..

Parameters:
    path        - path to images directory
    batch_size  - size of one batch
    
Return:
    
"""
def store_images(path, batch_size):
    image_paths = io.read_dir(path, ['jpg','jpeg','bmp','png','JPEG'])
    batch = []
    for i in range(len(image_paths)):
        # Load image
        image = load_image(image_paths[i])
        
        # Add image to batch
        batch.append(image)
        
        # Batch full -> export
        if (len(batch) == batch_size) | (i == (len(image_paths) - 1)):
            np_batch = np.asarray(batch)
            io.save_h5data(path + "/data_" + str(i/batch_size + 1) + ".h5", data = {'data':np_batch})
            batch = []
    
    return

"""
Create on Wed Dec 13

@author: anh_tt

Load images from directory to batch. Cache list image path as well

Parameters:
    path - path to images directory
    batch_size - size of one batch
    
Return:
    batch       - the first batch from cache
    image_paths - remaining image
"""
def load_batch(path, batch_size = 1, path_cache = []):
    batch = []
    
    if (len(path_cache) == 0):
        # Already run throught epoch -> Refill cache
        path_cache = io.read_dir(path, ['jpg','jpeg','bmp','png','JPEG'])
        
    for image_path in path_cache:
        #Load image 
        image = load_image(image_path)
        
        # Add image to batch
        batch.append(image)
        
        # Remove image which been added from cache
        path_cache.remove(image_path)
        
        # Batch full or cache is empty -> break
        if (len(batch) == batch_size) | (len(path_cache) == 0):
            break
    
    return (np.asarray(batch), path_cache)
    
# FOR TEST ONLY
