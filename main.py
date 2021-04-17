
import os
import sys
from pathlib import Path
from random import seed
from random import randint
from time import time
import argparse
import numpy as np
from utils.args_parsing import args_parsing
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from utils.data_loader import load_data
from custom_layers.nufft_nn import create_nufft_nn


def main(args):

    directory = Path('./saved_predictions/')
    directory.mkdir(exist_ok=True)
    directory = Path('./saved_models/')
    directory.mkdir(exist_ok=True)
    directory = Path('./training_checkpoints/')
    directory.mkdir(exist_ok=True)
    conv_layers = args.conv_layers
    lstm_layers = args.lstm_layers
    input_yx_size = tuple(args.input_yx_size)
    flatten = args.is_flatten_gt
    is_flat_channel_in = args.is_flat_channel_in
    #dense layers 
    dense_layers = args.dense_layers

    #transpose convultional layers of the Decoder
    trans_conv_layers = args.trans_conv_layers
    padding = {}

    for ix in args.padding:
        padding[ix] = True
    #sizes of the kernels
    kernel_sizes = {}
    if len(args.kernel_sizes_decoder)%3 !=0:
        sys.exit("Error input kernel size of decoder, the \
            input shoud be of this form: #index of layer #kernel width #kernel height")
    for ix in range(0,len(args.kernel_sizes_decoder),3):
        i = args.kernel_sizes_decoder[ix]
        kernel_sizes[i] = (args.kernel_sizes_decoder[ix+1],
            args.kernel_sizes_decoder[ix+2])

    #sizes of conv strides
    strides_decoder = {}
    
    if len(args.strides_decoder)%3 !=0:
        sys.exit("Error input stride size of conv Layers, the \
            input shoud be of this form: #index of layer #stride width #stride height")
    for ix in range(0,len(args.strides_decoder),3):
        i =  args.strides_decoder[ix]
        strides_decoder[i] = (args.strides_decoder[ix+1],
            args.strides_decoder[ix+2])

    #Upsampling window size
    upsampling_layers = {}

    if len(args.upsampling_layers)%3 !=0:
        sys.exit("Error input window size of upsmapling Layers, the \
            input shoud be of this form: #index of layer #kernel width #kernel height")
    for ix in range(0,len(args.upsampling_layers),3):
        i = args.upsampling_layers[ix]
        upsampling_layers[i] = (args.upsampling_layers[ix+1],
            args.upsampling_layers[ix+2])


    #Encoder Architecture
    #sizes of the kernels
    kernel_sizes_encoder = {}

    if len(args.kernel_sizes_encoder)%3 !=0:
        sys.exit("Error input kernel size of encoder, the \
            input shoud be of this form: #index of layer #kernel width #kernel height")
    for ix in range(0,len(args.kernel_sizes_encoder),3):
        i = args.kernel_sizes_encoder[ix]
        kernel_sizes_encoder[i] = (args.kernel_sizes_encoder[ix+1],
            args.kernel_sizes_encoder[ix+2])



    #sizes of conv strides
    strides_encoder = {}
    if len(args.strides_encoder)%3 !=0:
        sys.exit("Error input stride size of conv Layers, the \
            input shoud be of this form: #index of layer #stride width #stride height")
    for ix in range(0,len(args.strides_encoder),3):
        i = args.strides_encoder[ix]
        strides_encoder[i] = (args.strides_encoder[ix+1],
            args.strides_encoder[ix+2])

    #sizes of pooling windows
    pooling_layers = {}

    if len(args.pooling_layers)%3 !=0:
        sys.exit("Error input window size of pooling layers, the \
            input shoud be of this form: #index of layer #window width #window height")
    for ix in range(0,len(args.pooling_layers),3):
        i = args.pooling_layers[ix]
        pooling_layers[i] = (args.pooling_layers[ix+1],
            args.pooling_layers[ix+2])

    #tuple for batch normalization, if the first tuple 

    is_batchnorm = (args.is_batchnorm_dense, args.is_batchnorm_conv)
    dropout = args.dropout
    reg = args.reg
    flatten = args.is_flatten_gt

 


    #optimization settings
    is_CuDNNLSTM = args.is_CuDNNLSTM
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    num_test_samples = args.num_test_samples
    save_weights = args.save_weights
    load_weights = args.load_weights
    num_samples = args.num_samples
    save_train_prediction = args.save_train_prediction
    save_test_prediction = args.save_test_prediction
    verbose = args.verbose
    validation_ratio = args.validation_ratio
    decay = args.decay

    #initialize Encoder's arch
    """
    input_shape, conv_intermediate_dims,strides ={}, kernel_sizes={},pooling_layers={},
            is_batchnorm = False,dropout = 0.0,reg=0.0
    """
    encoderArch = [input_yx_size, conv_layers, strides_encoder, kernel_sizes_encoder, 
        pooling_layers, is_batchnorm[1],dropout,reg ]
    #initialize Decoder's arch
    """
    dense_intermediate_dims,conv_intermediate_dims,input_shape,strides= {},kernel_sizes={},upsampling_layers ={},
            is_flatten = False, is_batchnorm = (False,False),dropout = 0.0,reg=0.0

    """
    decoderArch = [dense_layers,trans_conv_layers,lstm_layers[len(lstm_layers)-1], strides_decoder,
        kernel_sizes, upsampling_layers,padding ,flatten, is_batchnorm, dropout, reg]



    #declare and initialize the network
    model = create_nufft_nn(encoderArch,lstm_layers,decoderArch,input_yx_size,is_CuDNNLSTM)

    opt = tf.keras.optimizers.Adam(lr=learning_rate, decay=decay)


    loss = tf.keras.losses.MeanSquaredError()
    mertric = ['mse']
    if args.loss is "MAE":
        loss = tf.keras.losses.MeanAbsoluteError()
        mertric = ['mae']


    model.compile(
        loss= loss,
        optimizer=opt,
        metrics=mertric,
    )


    y_axis_len,x_axis_len = input_yx_size
    input_shape= (None,y_axis_len,x_axis_len)
    model.build(input_shape=input_shape)

    model.summary()
    if load_weights :
        model.load_weights('./training_checkpoints/cp-best_loss.ckpt')

    #edit data_loader.py if you want to play with data

    input_ks, ground_truth = load_data(num_samples,is_flatten= flatten,is_flat_channel_in=is_flat_channel_in)
    input_ks_max = np.max(input_ks)
    input_ks = input_ks / input_ks_max

    
    checkpoint_path = "training_checkpoints/cp-best_loss.ckpt"


    cp_callback = []

    NAME = "NUFFT_NET"

    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
    cp_callback.append(tensorboard)
    if save_weights:
        cp_callback.append( tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True,
                                                        save_weights_only=True,
                                                        verbose=verbose))
    if args.is_train:
        model.fit(input_ks,ground_truth,batch_size = batch_size,epochs=epochs,
                validation_split=validation_ratio,callbacks = cp_callback)

    if args.name_model != "":
        print("Saving The Model.. ")
        model.save('./saved_models/'+args.name_model)

    dict_name = './saved_predictions/'
    #return to image size
    x_axis_len = int(x_axis_len/4)
    np.random.seed(int(time()))

    if save_train_prediction <= num_samples:
        rand_ix = np.random.randint(0,num_samples-1,save_train_prediction)

        kspace = input_ks[rand_ix]
        if args.save_input:
            
            np.save("./saved_predictions/inputs.npy",input_ks[rand_ix] * input_ks_max)
        ground_truth = ground_truth[rand_ix]
        preds = model.predict(kspace,batch_size=save_train_prediction)
        for i in range(save_train_prediction):
            

            output = np.reshape(preds[i], (y_axis_len*2, x_axis_len))
            output = output * 255
            output[np.newaxis,...]
            output_gt = ground_truth[i]
            output_gt[np.newaxis,...]
            if flatten:
                output_gt = np.reshape(output_gt, (y_axis_len*2, x_axis_len))
            output = np.concatenate([output,output_gt],axis=0)
            np.save(dict_name+'prediction%d.npy'%(i+1),output)


        input_ks, ground_truth = load_data(num_test_samples,'test',is_flatten= flatten,is_flat_channel_in=is_flat_channel_in)
        input_ks_max  = np.max(input_ks)
        input_ks = input_ks / input_ks_max
    if args.is_eval:
        model.evaluate(input_ks,ground_truth,batch_size,verbose,callbacks=cp_callback)

    if save_test_prediction <= num_test_samples:
        rand_ix = np.random.randint(0,num_test_samples-1,save_test_prediction)
        kspace = input_ks[rand_ix]
        if args.save_input:
            np.save("./saved_predictions/test_inputs.npy",input_ks[rand_ix] * input_ks_max)
        ground_truth = ground_truth[rand_ix]
        preds = model.predict(kspace,batch_size=save_test_prediction)
        for i in range(save_test_prediction):
            

            output = np.reshape(preds[i], (y_axis_len*2, x_axis_len))
            output = output * 255
            output[np.newaxis,...]
            output_gt = ground_truth[i]
            output_gt[np.newaxis,...]
            if flatten:
                output_gt = np.reshape(output_gt, (y_axis_len*2, x_axis_len))
            output = np.concatenate([output,output_gt],axis=0)
            np.save(dict_name+'test_prediction%d.npy'%(i+1),output)



if __name__ == '__main__':
    args = args_parsing()
    main(args)
