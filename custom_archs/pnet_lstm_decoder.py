import numpy as np
import os
import sys
sys.path.append(".")
import argparse
from pathlib import Path

from time import time
import tensorflow as tf
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers import Input 
from tensorflow.python.keras.layers import Dense, Reshape
from tensorflow.python.keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from tensorflow.python.keras.layers import Lambda, concatenate
from utils.data_loader import load_data
from utils.args_parsing import args_parsing
from tensorflow.python.keras.callbacks import TensorBoard

from tensorflow.python.keras.layers import Dense, Conv2D, Conv2DTranspose,Reshape, Dropout, CuDNNLSTM
from tensorflow.python.keras import  models, Model

# pylint: disable=unnecessary-semicolon

def mat_mul(A, B):
    return tf.matmul(A, B)

def exp_dim(global_feature, num_points):
    return tf.tile(global_feature, [1, num_points, 1])


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
    is_flat_channel_in = args.is_flat_channel_in
    input_points = Input(shape=(num_points, 4))

    x = input_points
    x = Convolution1D(64, 1, activation='relu',
                    input_shape=(num_points, 4))(x)
    x = BatchNormalization()(x)
    x = Convolution1D(128, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(512, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=num_points)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(16, weights=[np.zeros([256, 16]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,0, 0, 0, 0,1]).astype(np.float32)])(x)
    input_T = Reshape((4, 4))(x)

    # forward net
    g = Lambda(mat_mul, arguments={'B': input_T})(input_points)
    g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
    g = BatchNormalization()(g)

    # feature transformation net
    f = Convolution1D(64, 1, activation='relu')(g)
    f = BatchNormalization()(f)
    f = Convolution1D(128, 1, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Convolution1D(128, 1, activation='relu')(f)
    f = BatchNormalization()(f)
    f = MaxPooling1D(pool_size=num_points)(f)
    f = Dense(512, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(256, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
    feature_T = Reshape((64, 64))(f)

    # forward net
    g = Lambda(mat_mul, arguments={'B': feature_T})(g)
    seg_part1 = g
    g = Convolution1D(64, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(32, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(32, 1, activation='relu')(g)
    g = BatchNormalization()(g)

    # global_feature
    global_feature = MaxPooling1D(pool_size=num_points)(g)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)

    # point_net_seg
    c = concatenate([seg_part1, global_feature])
    """ c = Convolution1D(512, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(256, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 1, activation='relu')(c)
    c = BatchNormalization()(c) """
    c = Convolution1D(256, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 4, activation='relu',strides=4)(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 4, activation='relu',strides=4)(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 4, activation='relu',strides=4)(c)
    c = BatchNormalization()(c)
    c = Convolution1D(64, 4, activation='relu',strides=4)(c)
    c = BatchNormalization()(c)
    c = Convolution1D(64, 4, activation='relu',strides=4)(c)
    c = BatchNormalization()(c)
    c = Convolution1D(32, 1, activation='relu')(c)
    c = BatchNormalization()(c)  
    """ c = Convolution1D(128, 4, activation='relu',strides=4)(c)
    c = Convolution1D(64, 4, activation='relu',strides=4)(c)
    c = Convolution1D(32, 4, activation='relu',strides=4)(c)
    c = Convolution1D(16, 1, activation='relu')(c)
    c = Convolution1D(1, 1, activation='relu')(c) """
    #c = tf.keras.backend.squeeze(c,3);
    c = CuDNNLSTM(64, return_sequences=False)(c)
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
    prediction =tf.keras.layers.Reshape([ 512, 256])(c)
    """ c1 ,c2  = tf.split(c,[256,256],axis=1,name="split")
    complexNum = tf.dtypes.complex(
        c1,
        c2,
        name=None
    )

    complexNum =tf.signal.ifft2d(
        complexNum,
        name="IFFT"
    )
    real = tf.math.real(complexNum)
    imag = tf.math.imag(complexNum)

    con = concatenate([real,imag])

    prediction  =tf.keras.layers.Reshape([ 512, 256])(con)
    """
    # define model
    model = Model(inputs=input_points, outputs=prediction)
    opt = tf.keras.optimizers.Adam(lr=learning_rate ,decay=decay)

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
 
    model.summary()
    if load_weights :
        model.load_weights('./training_checkpoints/cp-best_loss.ckpt')

    #edit data_loader.py if you want to play with data
    input_ks, ground_truth = load_data(num_samples,is_flat_channel_in=is_flat_channel_in)

    input_ks = input_ks / np.max(input_ks)

    checkpoint_path = "./training_checkpoints/cp-{epoch:04d}.ckpt"
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
        model.save('./saved_mdoels/'+args.name_model)
    dict_name = './saved_predictions/'
    #return to image size
    x_axis_len = int(x_axis_len/4)
    np.random.seed(int(time()))

    if save_train_prediction <= num_samples:
        rand_ix = np.random.randint(0,num_samples-1,save_train_prediction)
        #kspace = np.zeros((save_train_prediction,
                            #y_axis_len,input_ks[rand_ix].shape[1]))
        kspace = input_ks[rand_ix]
        if args.save_input:
            np.save("./saved_predictions/inputs.npy",input_ks[rand_ix])
        ground_truth = ground_truth[rand_ix]
        preds = model.predict(kspace,batch_size=save_train_prediction)
        for i in range(save_train_prediction):
            

            output = np.reshape(preds[i], (y_axis_len*2, x_axis_len))
            output = output * 255
            output[np.newaxis,...]
            output_gt = ground_truth[i]
            output_gt[np.newaxis,...]
            output = np.concatenate([output,output_gt],axis=0)
            np.save(dict_name+'prediction%d.npy'%(i+1),output)

        input_ks, ground_truth = load_data(num_test_samples,'test',is_flat_channel_in=is_flat_channel_in)

        input_ks = input_ks / np.max(input_ks)
    if args.is_eval:
        model.evaluate(input_ks,ground_truth,batch_size,verbose,callbacks=cp_callback)

    if save_test_prediction <= num_test_samples:
        rand_ix = np.random.randint(0,num_test_samples-1,save_test_prediction)
        kspace = input_ks[rand_ix]
        if args.save_input:
            np.save("./saved_predictions/test_inputs.npy",input_ks[rand_ix])
        ground_truth = ground_truth[rand_ix]
        preds = model.predict(kspace,batch_size=save_test_prediction)
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
