# PointNet LSTM Convultional Decoder

Testd on Pytorch 1.3 Python 3.6, Ubuntu 16.4, and 18.04.

Decoder base on PointNet++ [1]. LSTM, and convultional decoder with 20 layers.

There multiple versions:

tain_pnet.py uses the original pointnet++ with feature propagation layer.
tain_pnet_modified.py uses the modified pointnet++less parameters with feature propagation layer.

tain_pnet_modified.py uses the modified pointnet++less parameters with feature propagation layer,
and convlutional lstm.

IMPORTANT: These architecutre are hard coded for dataset generated from images256x256

Validation dataset is not recommended, due to a bug I could not solve.
 thus assgin samples for for test data as the number of GPUs.
 Later you can only do model evaluation with --skip_train True, and assign as many number of test samples.

Best Result achieved so far : MSE loss of 0.09

 # Data
    The inputs are of shape [batch, 2, 3 , num_points]
    num_points is number of samples points in kspace, which corresponds 
    to image_height * images_width.
    axis= 1 is for imaginar and real part(feuture), and x and y coordinates(coordiantes).
    axis= 2 devides between imaginary and real part of feuture part.
    and x and y coordinates of coordinates part.
    All of which have a vector of ones added making axis=2 of size 3.
    

 # list of Commands 

    --batch_size, type=int, default=32, help=input batch size
    --workers, type=int, default=4, help=number of data loading workers
    --epoch, type=int, default=201, help=number of epochs for training
    --pretrain, type=str, default=None,help=whether use pretrain model
    --gpu, type=str, default=0, help=specify gpu device
    --learning_rate, type=float, default=0.001, help=learning rate for training
    --decay, type=float, default=1e-4, help=weight decay
    --optimizer, type=str, default=Adam, help=type of optimizer
    --multi_gpu, type=str, default=None, help=whether use multi gpu training
    --jitter, default=False, help="randomly jitter point cloud"
    --step_size, type=int, default=20, help="randomly rotate point cloud"
    "--numTrainSamples",type=int, default=21118, help="number of tranining samples"
    "--numTestSamples",type=int, default=2300, help="number of test samples"
    --model_name, type=str, default=pointnet2, help=Name of model
    "--local_rank", default=0, type=int
    "--skip_training", default=False, type=bool


Note: 
--pretrain is not functioning.





















[1]
Qi, C. R., Yi, L., Su, H., & Guibas, L. J. 2017. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. In Advances in neural information processing systems pp. 5099-5108.

