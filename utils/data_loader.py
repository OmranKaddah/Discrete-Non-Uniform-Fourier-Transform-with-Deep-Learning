from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.image as mpimg
import os
from PIL import Image


def load_data(num_samples,cat = 'train', is_flatten = False,is_flat_channel_in= False):
    """
    Arguments:
        num_samples: int
            pass the number of samples to be loaded.
        
        cat: string
            specify whether the loaded data is for training
            and validation, or for test phase.
        
        is_flatten: bool
            specify whether the ground truth will a flatt array.
            This usually used if the last layer in the network
            is a dense fully connected layer.
        
        is_flat_channel_in: bool
            specify whether the input is to be divede
            into 4 channels, and each channel is falt.
            The flat input is real, imaginary,
            x_axis coordinate, and y axis  channels.

    Returns:
        input_ks, ground_truth: numpy array tuple
            input in kaspace and ground truth.
            Both are tensors of which the fist axis
            is the smaple index. The input will have
            3 dimension if is_flat_channel_in is False,
            and  else 4 dimensions.
            The output will have 2 dimension if
            is_is_flatten is not True.
            Of shape will be (2*Y axis_len,xaxis len)
            Otherwise the ground truth 1 dimension
            with shape 2*Y axis_len * xaxis len



    """

    ypath='./dataset/ground_truth/train/'
    xpath='./dataset/input/train/'
    
    input_ks = []
    ground_truth = []

    input_ks = []
    ground_truth = []
    num_samples_range = range(1,num_samples+1)
    permuted_idx = np.random.permutation(num_samples_range)
    
    if cat is 'train' or cat is 'validation':
        for i in permuted_idx:
        #{
            name  = xpath +   str(i) + '.npy'
            #x images: input images
            temp = np.load(name)
            # makes the input flat, divide real, imaginary,
            #x_axis coordinate, and y axis into channels.
            # that is the input will have shape:
            # the output is matrix/
            if is_flat_channel_in:
                temp = np.reshape(temp,(temp.shape[0]*int(temp.shape[1]/4), 4))

            input_ks.append(np.asarray(temp))
            
            #y images: ground truth
            name = ypath + str(i)+ '.npy'
            temp = np.load(name) 
        

            if is_flatten:
                temp = temp.flatten()

            ground_truth.append(temp)
          
    else :
        xpath='./dataset/input/test/'
        ypath='./dataset/ground_truth/test/'
        for i in range(1,num_samples+1):
       
            name  = xpath +   str(i) + '.npy'
            #x images: input images
            temp = np.load(name)

            if is_flat_channel_in:
                temp = np.reshape(temp,(temp.shape[0]*int(temp.shape[1]/4), 4))

            input_ks.append(np.asarray(temp))

            name = ypath + str(i)+ '.npy'
            temp = np.load(name)  

            if is_flatten:
                temp = temp.flatten()

            ground_truth.append(temp)
    
    return np.asarray(input_ks),np.asarray(ground_truth)
