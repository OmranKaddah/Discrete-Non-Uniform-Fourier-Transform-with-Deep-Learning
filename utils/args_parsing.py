import argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def args_parsing():
    #Command Line Inputs
    CLI=argparse.ArgumentParser()
    CLI.add_argument("--input_yx_size",  
                    nargs="+",
                    default= [256,4*256],
                    type=int)
    CLI.add_argument("--conv_layers",
                    nargs="*",
                    default= [],
                    type=int)
    CLI.add_argument("--lstm_layers",
                    nargs="*",
                    default= [256],
                    type=int)
    CLI.add_argument("--dense_layers",
                    nargs="*",
                    default= [],
                    type=int)
    CLI.add_argument("--trans_conv_layers",
                    nargs="*",
                    default= [],
                    type=int)
    CLI.add_argument("--padding",
                    nargs="*",
                    default= [],
                    type=int)
    CLI.add_argument("--kernel_sizes_encoder",
                    nargs="*",
                    default= [],
                    type=int)
    CLI.add_argument("--kernel_sizes_decoder",
                    nargs="*",
                    default= [],
                    type=int)
    CLI.add_argument("--pooling_layers",
                    nargs="*",
                    default= [],
                    type=int)
    CLI.add_argument("--upsampling_layers",
                    nargs="*",
                    default= [],
                    type=int)
    CLI.add_argument("--strides_encoder",
                    nargs="*",
                    default= [],
                    type=int)
    CLI.add_argument("--strides_decoder",
                    nargs="*",
                    default= [],
                    type=int)
    CLI.add_argument("--is_flatten_gt",
                    type=str2bool, 
                    nargs='?',
                    const=True, 
                    default=False)
    CLI.add_argument("--is_flat_channel_in",
                    type=str2bool, 
                    nargs='?',
                    const=True, 
                    default=False)
    CLI.add_argument("--is_batchnorm_conv",
                    type=str2bool, 
                    nargs='?',
                    const=True, 
                    default=False)
    CLI.add_argument("--is_batchnorm_dense",
                    type=str2bool, 
                    nargs='?',
                    const=True, 
                    default=False)
    CLI.add_argument("--dropout",
                    default=0.0,
                    type=float)
    CLI.add_argument("--reg",
                    default=0.0,
                    type=float)
    CLI.add_argument("--is_CuDNNLSTM",
                    type=str2bool, 
                    nargs='?',
                    const=True, 
                    default=True)
    CLI.add_argument("--save_weights",
                    type=str2bool, 
                    nargs='?',
                    const=True, 
                    default=False)
    CLI.add_argument("--load_weights",
                    type=str2bool, 
                    nargs='?',
                    const=True, 
                    default=False)
    CLI.add_argument("--is_train",
                    type=str2bool, 
                    nargs='?',
                    const=True, 
                    default=True)
    CLI.add_argument("--is_eval",
                    type=str2bool, 
                    nargs='?',
                    const=True, 
                    default=True)

    CLI.add_argument("--batch_size",
                    default=16,
                    type=int)
    CLI.add_argument("--epochs",
                    default=10,
                    type=int)
    CLI.add_argument("--learning_rate",
                    default= 5e-4,
                    type=float)
    CLI.add_argument("--momentum",
                    default=9e-1,
                    type=float)

    CLI.add_argument("--num_test_samples",
                    default=8,
                    type=int)
    CLI.add_argument("--num_samples",
                    default=8,
                    type=int)
    CLI.add_argument("--validation_ratio",
                    default=0.04,
                    type=float)
    CLI.add_argument("--save_train_prediction",
                    default=5,
                    type=int)
    CLI.add_argument("--save_test_prediction",
                    default=5,
                    type=int)
    CLI.add_argument("--save_input",
                        type=str2bool, 
                        nargs='?',
                        const=True, 
                        default=True)

    CLI.add_argument("--verbose",
                        type=str2bool, 
                        nargs='?',
                        const=True, 
                        default=False)
    CLI.add_argument("--loss",
                    default='MSE',
                    type= str)
    CLI.add_argument("--decay",
                    default= 1e-6,
                    type= float)
    #for load_model.py
    CLI.add_argument("--name_model",
                    default= "",
                    type= str)
    
    # parse the command line
    args = CLI.parse_args()

    return args