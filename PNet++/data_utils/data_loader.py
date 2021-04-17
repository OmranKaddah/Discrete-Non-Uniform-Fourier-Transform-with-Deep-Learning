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
def loadData(numTrain,NumTest,isFlattened = False,isFlattenedIn= False,isPic = False):

#{
    ypath='./datasets/y/train/';
    xpath='./datasets/x/train/';
    numberImages  =numTrain;
    trainInput = [];
    trainGroundTruth = [];

    testInput = [];
    testGroundTruth = [];
    if isPic:
        for i in range(1,numberImages+1):
        #{
            name  = xpath +   str(i) + '.jpg';
            #x images: input images
            img = Image.open(name);
            temp = img.copy(); 
            trainInput.append(np.asarray(temp));
            img.close();
            #y images: ground truth
            name = ypath + str(i)+ '.jpg';
            img = Image.open(name).convert('L');
            temp = img.copy(); 
            trainGroundTruth.append(np.asarray(temp));
            img.close();
        #}
    else:
        for i in range(1,numberImages+1):
        #{
            name  = xpath +   str(i) + '.npy';
            #x images: input images
            temp = np.load(name);
            if isFlattenedIn:

                temp = np.reshape(temp,(temp.shape[0]*int(temp.shape[1]/4), 4))
            trainInput.append(np.asarray(temp));
            
            #y images: ground truth
            name = ypath + str(i)+ '.npy';
            temp = np.load(name); 
        

            if isFlattened:
                temp = temp.flatten();

            trainGroundTruth.append(temp);
          
        #}

    xpath='./datasets/x/test/';
    ypath='./datasets/y/test/';
    numberImages  =NumTest;
    if isPic:
            name  = xpath +   str(i) + '.jpg';
            #x images: input images
            img = Image.open(name);
            temp = img.copy(); 
            testInput.append(np.asarray(temp));
            img.close();
            #y images: ground truth
            name = ypath + str(i)+ '.jpg';
            img = Image.open(name).convert('L');
            temp = img.copy(); 
            testGroundTruth.append(np.asarray(temp));
            img.close();
    else:
        for i in range(1,numberImages+1):
        #{
            name  = xpath +   str(i) + '.npy';
            #x images: input images
            temp = np.load(name);


            if isFlattenedIn:
                temp = np.reshape(temp,(temp.shape[0]*int(temp.shape[1]/4), 4))


            testInput.append(np.asarray(temp));

            name = ypath + str(i)+ '.npy';
            temp = np.load(name);
          

            if isFlattened:
                temp = temp.flatten();

            testGroundTruth.append(temp);
           
        #}
    
    return ([np.asarray(trainInput),np.asarray(trainGroundTruth)],[np.asarray(testInput),np.asarray(testGroundTruth)]); 
#}