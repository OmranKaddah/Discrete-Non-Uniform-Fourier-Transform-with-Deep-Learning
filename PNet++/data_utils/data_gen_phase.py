from os import listdir
from os.path import isfile, join
import numpy as np
#import cv2
import os
import sigpy as sp
from tqdm import tqdm
#import cupy as cp
import math
import sigpy.mri as spmri
from PIL import Image
from random import seed
from random import randint
import argparse
# pylint: disable=unnecessary-semicolon
CLI=argparse.ArgumentParser();
CLI.add_argument(
  "--cat",
  default='train',  
  type= str,
);
CLI.add_argument(
  "--num",
  default=1,  
  type= int,
);
#parse argument
args = CLI.parse_args();

def fieldmapGen(s, num=1):
    """Generate a 2D fieldmap of shape s

    Args:
       s: shape of the fieldmap
       num: number of fieldmaps to generate
    Returns:
        fieldmap

    """

    fieldmap = np.zeros((num,) + s.shape);

    for i in range(num):
        x_offset = np.random.rand(1) * 2 * np.pi  - 0.5;
        y_offset = np.random.rand(1) * 2 * np.pi  - 0.5;

        x_fac = np.random.rand(1) * 1.5;
        y_fac = np.random.rand(1) * 1.5;

        x = np.linspace( - np.pi, np.pi, num=s.shape[0]) * x_fac + x_offset;
        y = np.linspace( - np.pi, np.pi, num=s.shape[1]) * y_fac + y_offset;

        xx, yy  = np.meshgrid(x,y);

        fieldmap[i,:,:] = (np.sin( xx ) + np.sin( yy )).T;
        
    return fieldmap;
def fermat_spiral(size):
    data=[]    
    
    linespace = np.linspace(0,size/(2),size*size)
    #d=dot/linespace.shape[0]
    
    
    #d=dot/len(linespace)

    for i in linespace: 
     
        x = (i) * math.cos(i)
        y = (i) * math.sin(i)

        data.append([x,y])
    narr = np.array(data)
    
    return narr

seed(1);
readPath='./datasets/images/'+ args.cat +'/';
inputpath='./datasets/y/'+ args.cat +'/';
outputpath='./datasets/x/'+ args.cat +'/';
numberImages  =args.num;
xImgDim = 256;
yImgDim = 256;


X = np.linspace(-xImgDim/2,xImgDim/2,xImgDim);
Y = X;
xs, ys = np.meshgrid(X,Y);
trajCart = np.stack((ys,xs),axis=-1);
trajRad = spmri.radial((xImgDim,xImgDim,2), (xImgDim,yImgDim));
trajSpiral = fermat_spiral(256)
trajSpiral = np.reshape(trajSpiral,(yImgDim,xImgDim,2))

for i in tqdm(range(1,numberImages+1)):
    #{
    name  = readPath +   str(i) + '.jpg';
    image = Image.open(name).convert('L');
    arr = np.array(image);
    arr=arr/np.max(arr[...]);
    #simulate phase
    fieldMap  = fieldmapGen(arr)[0,:,:];
    #add phase information
    arr = arr * np.exp(1j*fieldMap);
    name  = inputpath+ str(i);
    groundTruth = np.concatenate((arr.real,arr.imag),axis=0);
    groundTruth = groundTruth.astype(np.float32);

    np.save(name,groundTruth);
    ## select random trajectory:
    value = randint(0, 2)
    if value is 0:
        traj = trajCart;
        kspace = np.fft.fft(arr)
    elif value is 1:
        traj = trajRad;
        np.random.shuffle(traj)
        #apply Non uniform Fourier Transform
        kspace = sp.nufft(arr, traj);
    else:
        traj = trajSpiral;
     
        #apply Non uniform Fourier Transform
        kspace = sp.nufft(arr, traj);

    xy = np.reshape(traj,(xImgDim*yImgDim,2));
    
    ones = np.ones((xImgDim*yImgDim,1));
    xy = np.concatenate([xy,ones],axis=1);
    xy = np.transpose(xy);
    xy = xy[np.newaxis,...];
    real = kspace.real.flatten();
    imag = kspace.imag.flatten();
    
    real = real[...,np.newaxis];
    imag = imag[...,np.newaxis];
    ones = np.ones(imag.shape);
    ri  = np.concatenate([real,imag,ones],1);
    ri = np.transpose(ri);
    ri = ri[np.newaxis,...];

    #construct the input matrix for that is used later for training.
    #inMat[X,Y,ones,real,imaginary,ones]
    inMat = np.concatenate((xy,ri),axis=0);
    inMat = inMat.astype(np.float32)
    name  = outputpath+ str(i);
    np.save(name,inMat);

    #}

print("\n DONE \n");


