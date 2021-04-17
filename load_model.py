
import os
import sys
from random import seed
from random import randint
from time import time
import argparse
import numpy as np
from PIL import Image
from utils.args_parsing import args_parsing
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from utils.data_loader import load_data
from custom_layers.nufft_nn import NUFFT_NN


def main(args):
    
    input_yx_size = tuple(args.input_yx_size)

    #optimization settings
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
    flatten = args.is_flatten_gt
    is_flat_channel_in = args.is_flatten_in
    y_axis_len,x_axis_len = input_yx_size
    decay = args.decay
    #declare and initialize the network
    model = tf.keras.models.load_model("saved_models/"+args.load_model)

    if args.recompile:
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
    model.summary()
 

    #edit data_loader.py if you want to play with data
    input_ks, ground_truth = load_data(num_samples)

    input_ks = input_ks / np.max(input_ks)

    checkpoint_path = "train_checkpoints/cp-{epoch:04d}.ckpt"
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

    if save_train_prediction <= num_samples:
        rand_ix = np.random.randint(0,num_samples-1,save_train_prediction)

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
