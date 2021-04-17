from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.image as mpimg
import os
from PIL import Image
# pylint: disable=unnecessary-semicolon

def rgb2gray(rgb):
    if rgb.ndim<3:
        return rgb;
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2];
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b;
    return gray;
def loadData(rangeIndex,numTrain = False,NumTest = False,isFlattened = False,isFlattenedIn= False):

#{
    ypath='./datasets/y/train/';
    xpath='./datasets/x/train/';
    Inputs = [];
    GroundTruth = [];


    if numTrain:

   
        for i in rangeIndex:
        #{
            name  = xpath +   str(i) + '.npy';
            #x images: input images
            temp = np.load(name);
            if isFlattenedIn:

                temp = np.reshape(temp,(temp.shape[0]*int(temp.shape[1]/4), 4))
            Inputs.append(np.asarray(temp));
            
            #y images: ground truth
            name = ypath + str(i)+ '.npy';
            temp = np.load(name); 
        

            if isFlattened:
                temp = temp.flatten();

            GroundTruth.append(temp);
          
        #}


    else:
        xpath='./datasets/x/test/';
        ypath='./datasets/y/test/';
        numberImages  =NumTest;

        for i in rangeIndex:
        #{
            name  = xpath +   str(i) + '.npy';
            #x images: input images
            temp = np.load(name);


            if isFlattenedIn:
                temp = np.reshape(temp,(temp.shape[0]*int(temp.shape[1]/4), 4))


            Inputs.append(np.asarray(temp));

            name = ypath + str(i)+ '.npy';
            temp = np.load(name);
          

            if isFlattened:
                temp = temp.flatten();

            GroundTruth.append(temp);
           
        #}
    
    return np.asarray(Inputs),np.asarray(GroundTruth); 
#}