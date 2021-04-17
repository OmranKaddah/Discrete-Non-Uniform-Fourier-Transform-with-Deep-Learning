import sys
sys.path.append(".")
from pathlib import Path

from random import randint
import numpy as np
from time import time
import os
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
import argparse

from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose,Reshape, CuDNNLSTM, Bidirectional
from tensorflow.python.keras import  models
from utils.data_loader import load_data
from utils.args_parsing import args_parsing



def main(args):
    directory = Path('./saved_predictions/')
    directory.mkdir(exist_ok=True)
    directory = Path('./saved_models/')
    directory.mkdir(exist_ok=True)
    directory = Path('./training_checkpoints/')
    directory.mkdir(exist_ok=True)
    #passed arguments
    input_yx_size = tuple(args.input_yx_size)
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    num_test_samples = args.num_test_samples
    save_weights = args.save_weights
    load_weights = args.load_weights
    every = args.every
    num_samples = args.num_samples
    save_train_prediction = args.save_train_prediction
    save_test_prediction = args.save_test_prediction
    verbose = args.verbose
    validation_ratio = args.validation_ratio
    y_axis_len,x_axis_len = input_yx_size
    val = input("Enter number of samples to load to GPUs: ") 
    gpu_load = int(val)
    strategy = tf.distribute.MirroredStrategy()



    BATCH_SIZE_PER_REPLICA = batch_size
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync


    with strategy.scope():
        model = models.Sequential()


        model.add(Bidirectional(CuDNNLSTM(256, return_sequences=True),input_shape=input_yx_size))
        model.add(CuDNNLSTM(256, return_sequences=True,input_shape=input_yx_size))
        model.add(tf.keras.layers.Reshape([ 256, 256,1]))
        model.add(Conv2DTranspose(64, (3,3),padding="same",activation="relu",strides=(2, 1)))
        model.add(Conv2DTranspose(64, (3,3),padding="same",activation="relu",strides=(1, 1)))
        model.add(Conv2DTranspose(256, (1,1),activation="relu",strides=(1, 1)))
        model.add(Conv2DTranspose(256, (1,1),activation="relu",strides=(1, 1)))
        model.add(Conv2DTranspose(64, (1,1),activation="relu",strides=(1, 1)))
        model.add(Conv2DTranspose(32, (1,1),activation="relu",strides=(1, 1)))
        model.add(Conv2DTranspose(1, (1,1),strides=(1, 1)))

        model.add(tf.keras.layers.Reshape([ 512,256]))





        opt = tf.keras.optimizers.Adam(lr=learning_rate, decay=1e-6)

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
    input_ks, ground_truth = load_data(num_samples)

    input_ks = input_ks / np.max(input_ks)

    checkpoint_path = "training_checkpoints/cp-best_loss.ckpt"


    cp_callback = []

    NAME = "NUFFT_NET"

    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
    cp_callback.append(tensorboard)
    if save_weights:
        cp_callback.append( tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                        save_best_only=True,
                                                        save_weights_only=True,
                                                        verbose=verbose))
    epochs_losses = [] 
    if args.is_train:
        for e in range(epochs):
            losses = []
            for bth, i in enumerate(range(0,num_samples,gpu_load)):
                #loss_batch = model.train_on_batch(input_ks[i:i+gpu_load],ground_truth[i:i+gpu_load])
                #losses.append(loss_batch)
                model.fit(input_ks[i:i+gpu_load],ground_truth[i:i+gpu_load],BATCH_SIZE
                        ,verbose=verbose,callbacks=cp_callback)
                #print("the %d'th batch in %d'th epoch  and is loss: %f"%(bth,e,loss_batch))
            #if i%every is 0:
                #model.save_weights("./training_checkpoints/cp-{}.ckpt".format(e))
            #avg_loss = sum(losses) / len(losses)
            #epochs_losses.append(avg_loss)
            #print("End of %d th Epoch and AVG loss is %f"%(e,avg_loss))
            print("End of Epoch {}".format(e))
   


    if args.name_model is not "":
        model.save('./saved_models/'+args.name_model)
        
    #____________________saving predicitons

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

        input_ks, ground_truth = load_data(num_test_samples,'test',is_flatten= flatten,is_flat_channel_in=is_flat_channel_in)

        input_ks = input_ks / np.max(input_ks)
    if args.is_eval:
        model.evaluate(input_ks,ground_truth,batch_size,verbose,callbacks=cp_callback)

    if save_test_prediction <= num_test_samples:
        rand_ix = np.random.randint(0,num_test_samples-1,save_test_prediction)
        #kspace = np.zeros((save_test_prediction,
                            #y_axis_len,input_ks[rand_ix].shape[1]))
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
