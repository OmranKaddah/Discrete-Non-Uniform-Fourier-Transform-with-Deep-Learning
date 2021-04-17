import argparse
import math
from random import seed
from random import randint
import os
from os import listdir
from os.path import isfile, join
from time import time
import numpy as np

import sigpy as sp
import sigpy.plot as plt
import sigpy.mri as spmri
from PIL import Image
from tqdm import tqdm
CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--cat",
  default='train',
  help=" category of the data, that is whether the \
      the generated data is for taining and validation \
          or the for test data. Pass train for the frist \
           category and test for the second." , 
  type= str,
)
CLI.add_argument(
  "--num",
  default=1,
  help="pass number of sameples to be generated",  
  type= int,
)

CLI.add_argument(
  "--image_hight",
  default=256, 
  help="pass image hight, for which NUFFT is going to applied",
  type= int,
)
CLI.add_argument(
  "--image_width",
  default=256,
  help="pass image width, for which NUFFT is going to applied",  
  type= int,
)

CLI.add_argument(
  "--FFT_input",
  default=False, 
  help="pass bool for whether FFT is going to be applied \
      for the input. This is a case where network has only to learn\
      to be interpolate transformation into cartsian system",  
  type= bool,
)
#parse argument
args = CLI.parse_args()

def fieldmapGen(s, num=1):
    """Generate a 2D fieldmap of shape s

    Args:
       s: shape of the fieldmap
       num: number of fieldmaps to generate
    Returns:
        fieldmap

    """

    fieldmap = np.zeros((num,) + s.shape)

    for i in range(num):
        x_offset = np.random.rand(1) * 2 * np.pi  - 0.5
        y_offset = np.random.rand(1) * 2 * np.pi  - 0.5

        x_fac = np.random.rand(1) * 1.5
        y_fac = np.random.rand(1) * 1.5

        x = np.linspace( - np.pi, np.pi, num=s.shape[0]) * x_fac + x_offset
        y = np.linspace( - np.pi, np.pi, num=s.shape[1]) * y_fac + y_offset

        xx, yy  = np.meshgrid(x,y)

        fieldmap[i,:,:] = (np.sin( xx ) + np.sin( yy )).T
        
    return fieldmap

def fermat_spiral(x_axis,y_axis):
    """
    Arguments:

        x_axis: int
            width of the trajectory
        y_axis: int
            height of the trajectory
    
    Returns:
        numpy array of the trajectory
        with shape [y_axis,x_axis,2]

    """
    
    linespace = np.linspace(0,x_axis/(2),x_axis*x_axis)

    xs = []
    ys = []
    for i in linespace: 
     
        xs.append(i * math.cos(i))

    linespace = np.linspace(0,y_axis/(2),y_axis*y_axis)        
    for i in linespace:    
        ys.append( i * math.sin(i))
        
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    xs = xs[...,np.newaxis]
    ys = ys[...,np.newaxis]
    narr = np.concatenate([xs,ys],axis=-1)
    narr = np.reshape(narr,(y_axis,x_axis,2))
    return narr

seed(time())
cat = args.cat
read_path='./dataset/images/'+ cat +'/'
input_path='./dataset/ground_truth/'+ cat +'/'
output_path='./dataset/input/'+ cat +'/'

num_images  =args.num
is_FFT = args.FFT_input
x_axis_len = args.image_width
y_axis_len = args.image_hight

# making the cartsian trajectory
list_xs = np.linspace(-x_axis_len/2,x_axis_len/2,x_axis_len)
list_ys = list_xs
xs, ys = np.meshgrid(list_xs,list_ys)

#cartsian trajectory
cart_traj = np.stack((ys,xs),axis=-1)

#radial trajectory
rad_traj = spmri.radial((y_axis_len,x_axis_len,2), (y_axis_len,x_axis_len))

#spiral trajectory
spiral_traj = fermat_spiral(x_axis_len,y_axis_len)

print("Generating...\n")
for i in tqdm(range(1,num_images+1)):
    #{
    name  = read_path +   str(i) + '.jpg'
    image = Image.open(name).convert('L')
    arr = np.array(image)
    arr=arr/np.max(arr[...])
    #simulate phase
    fieldMap  = fieldmapGen(arr)[0,:,:]
    #add phase information
    arr = arr * np.exp(1j*fieldMap)
    if is_FFT:
        arr = np.fft.fft(arr)

    name  = input_path+ str(i)


    #[2*y_axis_len, x_axis_len ] = concatenate in 0 axis
    groundTruth = np.concatenate((arr.real,arr.imag),axis=0)
    groundTruth = groundTruth.astype(np.float32)
    np.save(name,groundTruth)

    value = randint(0, 2)
    if value is 0:
        traj = cart_traj
        kspace = np.fft.fft(arr)
    elif value is 1:
        traj = spiral_traj
        kspace = sp.nufft(arr, traj)
    else:
        traj = rad_traj
        np.random.shuffle(traj)
        #apply Non uniform Fourier Transform
        kspace = sp.nufft(arr, traj)
    #apply Non uniform Fourier Transform

    x_k = traj[:,:,0].flatten()
    y_k = traj[:,:,1].flatten()
    real = kspace.real.flatten()
    imag = kspace.imag.flatten()

    real = real[...,np.newaxis]
    imag = imag[...,np.newaxis]
    x_k = x_k[...,np.newaxis]
    y_k = y_k[...,np.newaxis]
    #construct the input matrix for that is used later for training.
    #each of the arrays has dimension [x_axis_len*y_axis_len ,1]
    #results after concatenation [x_axis_len*y_axis_len ,4]
    inMat = np.concatenate((real,imag,x_k,y_k),axis=1)
    
    #reshaping to [y_axis_len, 4* x_axis_len]
    # content of row inthe array will be as: 
    # [real[ix],imag[ix],x_k[ix],y_k[ix], real[ix+1], image[ix+1] ...]
    
    inMat  = np.reshape(inMat,(kspace.shape[0],4*kspace.shape[1]))
    name  = output_path+ str(i)
    inMat = inMat.astype(np.float32)
    np.save(name,inMat)
    
print("DONE")



