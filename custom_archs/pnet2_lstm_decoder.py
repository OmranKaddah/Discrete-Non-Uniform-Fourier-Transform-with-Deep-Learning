import numpy as np
import os
import sys
sys.path.append(".")
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Reshape
from tensorflow.python.keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from tensorflow.python.keras.layers import Lambda, concatenate
from utils.data_loader import load_data
from utils.args_parsing import args_parsing
from tensorflow.python.keras.callbacks import TensorBoard
import argparse
from PIL import Image
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Conv2D, Conv2DTranspose,Reshape, Dropout, CuDNNLSTM, UpSampling2D, Conv1D#, CuDNNLSTM
from tensorflow.python.keras import  models
from feutureProb import PointNetFeaturePropagation
from pNetSetAbst import PointNetSetAbstractionMsg



def main(arg):

    directory = Path('./saved_predictions/')
    directory.mkdir(exist_ok=True)
    directory = Path('./saved_models/')
    directory.mkdir(exist_ok=True)
    directory = Path('./training_checkpoints/')
    directory.mkdir(exist_ok=True)

    input_yx_size = tuple(args.input_yx_size)
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    num_test_samples = args.num_test_samples
    save_weights = args.save_weights
    every = args.every
    num_samples = args.num_samples
    save_train_prediction = args.save_train_prediction
    save_test_prediction = args.save_test_prediction
    verbose = args.verbose
    validation_ratio = args.validation_ratio
    y_axis_len,x_axis_len = input_yx_size
    decay = args.decay
    decay = args.decay
    load_weights = args.load_weights
    y_axis_len,x_axis_len = input_yx_size
    num_points = y_axis_len*x_axis_len 


    input_points = Input(shape=( 2, 2,num_points),batch_size=12)

    l0_xyz = input_points[:,0,:,:]
    print(l0_xyz.get_shape().as_list())
    values = input_points[:,1,:,:]
    l1_xyz, l1_points = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128],  [[32, 32, 64], [64, 64, 128], [64, 96, 128]])((l0_xyz, values))
    l2_xyz, l2_points = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], [[128, 128, 256], [128, 196, 256]])(l1_xyz, l1_points)
    #l3_xyz, l3_points = PointNetSetAbstractionM(npoint=None, radius=None, nsample=None, mlp=[256, 512, 1024], group_all=True)(l2_xyz, l2_points)
    l2_points = PointNetFeaturePropagation( mlp=[256, 256])(l1_xyz, l2_xyz, l1_points, l2_points)
    #l1_points = PointNetFeaturePropagation( mlp=[256, 128])(l1_xyz, l2_xyz, l1_points, l2_points)
    conv1 = Conv1D(128, 1)(l2_points)
    bn1 = tf.keras.activations.relu(BatchNormalization(conv1))
    drop1 = Dropout(0.5)(bn1)
    conv2 = tf.keras.activations.relu(Conv1D(64 , 1)())



    c = CuDNNLSTM(64, return_sequences=False)(conv2)
    #c =CuDNNLSTM(784, return_sequences=False))
    #c =CuDNNLSTM(256, return_sequences=False))

    #c = Reshape([16,16,1])(c)
    c = Reshape([8,8,1])(c)
    c = Conv2DTranspose(8, (3,3),padding="same",activation="relu",strides=(2,2))(c)
    c = Conv2DTranspose(8, (3,3),padding="valid",activation="relu")(c)
    #c =Dropout(0.4))
    c = tf.keras.layers.BatchNormalization()(c)
    c = Conv2DTranspose(16, (3,3),padding="valid",activation="relu")(c)
    #c =Dropout(0.4))
    c = tf.keras.layers.BatchNormalization()(c)
    c = Conv2DTranspose(32, (3,3),padding="valid",activation="relu")(c)
    #c =Dropout(0.4))
    c = tf.keras.layers.BatchNormalization()(c)
    c = Conv2DTranspose(32, (3,3),padding="valid",activation="relu")(c)
    #c =Dropout(0.4))
    c = tf.keras.layers.BatchNormalization()(c)
    c = Conv2DTranspose(32, (3,3),padding="valid",activation="relu")(c)
    #c =Dropout(0.4))
    c = tf.keras.layers.BatchNormalization()(c)
    c =Conv2DTranspose(64, (3,3),padding="valid",activation="relu")(c)
    #c =Dropout(0.4)) 
    c =tf.keras.layers.BatchNormalization()(c)
    c =Conv2DTranspose(64, (3,3),padding="valid",activation="relu")(c)
    #c =Dropout(0.4))
    c =tf.keras.layers.BatchNormalization()(c)

    #c =Dropout(0.4))


    c =Conv2DTranspose(128, (3,3),padding="same",activation="relu",strides=(2,2))(c)
    c =tf.keras.layers.BatchNormalization()(c)

    c =Conv2DTranspose(128, (3,3),padding="valid",activation="relu")(c)

    #c =Dropout(0.4))
    c =tf.keras.layers.BatchNormalization()(c)
    c =Conv2DTranspose(128, (3,3),padding="same",activation="relu",strides=(2,2))(c)
    c =tf.keras.layers.BatchNormalization()(c)

    c =Conv2DTranspose(128, (3,3),padding="valid",activation="relu")(c)
    c =tf.keras.layers.BatchNormalization()(c)

    #c =Dropout(0.4))
    #c =tf.keras.layers.BatchNormalization())
    c =Conv2DTranspose(64, (3,3),padding="same",strides=(4,2))(c)
    c =tf.keras.layers.BatchNormalization()(c)

    c =Conv2DTranspose(32, (3,3),padding="valid",activation="relu")(c)
    c =tf.keras.layers.BatchNormalization()(c)

    c =Conv2DTranspose(32, (3,3),padding="valid",activation="relu")(c)
    c =tf.keras.layers.BatchNormalization()(c)

    #c =Dropout(0.4))
    c =Conv2DTranspose(32, (3,3),padding="same",activation="relu",strides=(1, 1))(c)
    c =tf.keras.layers.BatchNormalization()(c)

    c =Conv2DTranspose(32, (3,1),padding="valid",activation="relu")(c)
    c =tf.keras.layers.BatchNormalization()(c)

    c =Conv2DTranspose(32, (3,1),padding="valid",activation="relu")(c)
    c =tf.keras.layers.BatchNormalization()(c)
    c =Conv2DTranspose(16, (1,1),padding="valid",activation="relu")(c)
    c =tf.keras.layers.BatchNormalization()(c)
    c =Conv2DTranspose(8, (1,1),padding="valid",activation="relu")(c)
    c =tf.keras.layers.BatchNormalization()(c)

    c =Conv2DTranspose(1, (1,1),padding="valid")(c)
    """ c =Conv2DTranspose(4, (1,1),padding="same",activation="relu"))
    c =Conv2DTranspose(2, (1,1),padding="same",activation="relu"))
    #c =Dropout(0.4))
    c =Conv2DTranspose(1, (1,1),padding="same")) """
    c =tf.keras.layers.Reshape([ 512, 256])(c)

    prediction = c


    # define model
    nufftNN = Model(inputs=input_points, outputs=prediction)
    opt = tf.keras.optimizers.Adam(lr=learningRate ,decay=2e-6);


    nufftNN.compile(
        loss='mean_squared_error',
        metrics=['mse'],
        #loss=SSIM,
        optimizer=opt

    );
    nufftNN.summary()

    opt = tf.keras.optimizers.Adam(lr=learning_rate ,decay=decay)

    loss = tf.keras.losses.MeanSquaredError()
    mertric = tf.keras.metrics.MeanSquaredError()
    if args.loss is "MAE":
        loss = tf.keras.losses.MeanAbsoluteError()
        mertric = tf.keras.metrics.MeanAbsoluteError()


    model.compile(
        loss= loss,
        optimizer=opt,
        metrics=mertric,
    )
 
    model.summary()
    if load_weights is not '':
        model.load_weights('training_checkpoints/cp-'+load_weights+'.ckpt')

    #edit data_loader.py if you want to play with data
    input_ks, ground_truth = load_data(num_samples)

    input_ks = input_ks / np.max(input_ks)

    checkpoint_path = "training_checkpoints/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create checkpoint callback
    #do you want to save the model's wieghts? if so set this varaible to true

    cp_callback = []

    NAME = "NUFFT_NET"

    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
    cp_callback.append(tensorboard)
    if save_weights:
        cp_callback.append( tf.keras.callbacks.ModelCheckpoint(checkpoint_dir,
                                                        save_weights_only=True,
                                                        verbose=verbose,period=every))

    if args.is_train:
        model.fit(input_ks,ground_truth,batch_size = batch_size,epochs=epochs,
                    validation_split=validation_ratio,callbacks = cp_callback)
    
    if args.name_model is not "":
        model.save('saved_mdoels/'+args.name_model)
    dict_name = 'saved_predictions/'
    #return to image size
    x_axis_len = int(x_axis_len/4)
    np.random.seed(int(time()))

    rand_ix = np.random.randint(0,num_samples-1,save_train_prediction)
    kspace = np.zeros((save_train_prediction,
                        input_ks[rand_ix].shape[0],input_ks[rand_ix].shape[1]))
    kspace = input_ks[rand_ix]
    ground_truth = ground_truth[rand_ix]
    preds = model(kspace,training=False)
    for i in range(save_train_prediction):
        

        output = np.reshape(preds[i], (y_axis_len*2, x_axis_len))
        output = output * 255
        output[np.newaxis,...]
        output_gt = ground_truth[i]
        output_gt[np.newaxis,...]
        output = np.concatenate([output,output_gt],axis=0)
        np.save(dict_name+'prediction%d.npy'%(i+1),output)

    input_ks, ground_truth = load_data(num_test_samples,'test',is_flat_channel_in=True)

    input_ks = input_ks / np.max(input_ks)
    model.evaluate(input_ks,ground_truth,batch_size,verbose,callbacks=cp_callback)

    rand_ix = np.random.randint(0,num_samples-1,save_test_prediction)
    kspace = np.zeros((save_test_prediction,
                        input_ks[rand_ix].shape[0],input_ks[rand_ix].shape[1]))
    kspace = input_ks[rand_ix]
    ground_truth = ground_truth[rand_ix]
    preds = model.predict(kspace,batch_size=1)
    for i in range(save_test_prediction):
        

        output = np.reshape(preds[i], (y_axis_len*2, x_axis_len))
        output = output * 255
        output[np.newaxis,...]
        output_gt = ground_truth[i]
        output_gt[np.newaxis,...]
        output = np.concatenate([output,output_gt],axis=0)
        np.save(dict_name+'test_prediction%d.npy'%(i+1),output)
        

if __name__ == '__main__':
    args = args_parsing()
    main(args)

