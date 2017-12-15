#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:12:55 2017

@author: anh_tt
"""

import os
import h5py
import numpy as np
import glob

"""
Create on Wed Dec 13

@author: anh_tt

Read *.h5 file base on keys

Parameters:
    path - path to h5 file 
    keys - a list of key in h5 file 
"""
def read_h5data(path, keys = []):
    result = {}
    with h5py.File(path, 'r') as hf:
        for key in keys:
            result[key] = np.array(hf.get(key))

    return result
        
"""
Create on Wed Dec 13

@author: anh_tt

Write array to *.h5 file

Parameters:
    path - path to directory you want to read
    data - a dictionary with key - value of data
"""        
def save_h5data(path, data = {}):
    save_path = os.path.join(os.getcwd(), path)
    with h5py.File(save_path, 'w') as hf:
        for (key, value) in data.items():
            hf.create_dataset(key, data = value)
    
    return 
        
"""
Create on Wed Dec 13

@author: anh_tt

Read all file path in a directory base on file extensions

Parameters:
    path        - path to directory you want to read
    extensions  - a list of file extension you want to read
"""        
def read_dir(path, extensions = []):
    data_path = []
    data_dir = os.getcwd()
    
    print(os.path.join(data_dir + path, "*.png"))
    
    for extension in extensions:
        data_path = data_path + glob.glob(os.path.join(data_dir + path, "*." + extension))
    
    return data_path


# FOR TESTING ONLY
#print(read_h5data("/../test/data_1.h5", ["data"]))
