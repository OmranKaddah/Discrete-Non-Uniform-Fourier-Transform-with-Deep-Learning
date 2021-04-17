from PIL import Image
from matplotlib import pyplot
import numpy as np
import argparse

CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--cat",
  default='train_validation',
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
args = CLI.parse_args()
prefix = ""
cat =  args.cat
if cat is "test":
    prefix = "test_"

   

#parse argument
args = CLI.parse_args()

y_axis_len = args.image_hight
x_axis_len = args.image_width
num_predicitions = args.num

#preds: list
#   tensors which has it first dimension
#   of size 2. The first index is for prediction
#   and the second is for ground truth.
preds = []
for i in range(num_predicitions):
    name = "saved_predictions/" + \
	        prefix +"prediction" + str(i+1) + ".npy"

    preds.append(np.load(name))

for i in range(num_predicitions):

    #processing the predection
    output = preds[i][0:2*y_axis_len]
    
    arrReal =np.copy( output[0:y_axis_len,:])
    arrImag =np.copy( output[y_axis_len:y_axis_len*2,:])
    #extracting the magnitude
    yy = np.sqrt(np.square(arrImag) + np.square(arrReal)) 

    im = Image.fromarray(np.uint8(yy))

    #for ploting
    pyplot.subplot(121,title='prediction')
    pyplot.imshow(im ,cmap='gray', vmin=0, vmax=255)
    #end

    #processing the ground truth
    output = preds[i][2*y_axis_len:4*y_axis_len]
    arrReal =np.copy( output[0:y_axis_len,:])
    arrImag =np.copy( output[y_axis_len:y_axis_len*2,:])
    #extracting the magnitude
    yy = np.sqrt(np.square(arrImag) + np.square(arrReal)) 
    yy = yy * 255
    im = Image.fromarray(np.uint8(yy))
    pyplot.subplot(122,title='ground truth')
    pyplot.imshow(im, cmap='gray', vmin=0, vmax=255)
    pyplot.show()
    #end
