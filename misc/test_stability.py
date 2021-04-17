from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import os
import sigpy as sp
#import cupy as cp
import sigpy.plot as plt
import sigpy.mri as spmri
from PIL import Image
from matplotlib import cm

# pylint: disable=unnecessary-semicolon

def cartisian2D( traj_shape, densitySAxis = (1.,1.), densityScale = 1., dtype=np.float):
    '''
    img_shape : tuple of two ints
        shape of the resultant trajctory
    densitySAxis :  tuple of two ints
        density in of sample in y and x axis respectivly per densityScale
    densityScale : float
        controlles the dinsity in each aixs
    dtype:   numpy dtype
        type of the output
    '''
    imgDimY, imgDimX = traj_shape;
    imgDimY = imgDimY/2;
    imgDimX = imgDimX/2;
    denistyY, densityX = densitySAxis;
    if(densityScale>0):
        factor1 =  denistyY / densityScale;
        factor2 =  densityX / densityScale;
    else:
        #{
        print("WARNING! density scale should be bigger than zero.");
        return None;
        #}
    #m1 , m2 = np.meshgrid(np.arange(-imgDimX, imgDimX + factor2, factor2), np.arange(-imgDimY, imgDimY + factor1, factor1));
    m1 , m2 = np.meshgrid(np.arange(-imgDimX, imgDimX , factor2), np.arange(-imgDimY, imgDimY , factor1));

    
    y, x = m2.shape;
    mesh = np.zeros(( y,x,2));

    mesh[:,:,0] = m1;
    mesh[:,:,1] = m2;
    return mesh.astype(np.float);


name  = 'img.jpg';
image = Image.open(name).convert('L');

arr = np.array(image) + 1j;
traj  = cartisian2D(arr.shape,[1,1],1);
plt.ScatterPlot(traj, title='Trajectory');
image.close()
arr=arr/np.max(arr[...]);

print(traj.shape)

kspaceNUFFT = sp.nufft(arr, traj);

plt.ImagePlot(np.log(kspaceNUFFT), title='k-space data from NUFFT');


kspaceFFT  = sp.fft(arr);


plt.ImagePlot(np.log(kspaceFFT), title='k-space data from FFT')
print(kspaceFFT.shape);
print(kspaceNUFFT.shape);
sumNUFFT = np.sum(kspaceNUFFT);
sumFFT  = np.sum(kspaceFFT);
if(np.allclose(kspaceNUFFT,kspaceFFT,rtol= 10,atol = 10) and np.isclose(sumNUFFT,sumFFT,rtol= 50,atol = 50)):
    print('Outputs are  similar!')
else:
    print('Outputs are NOT similar!')