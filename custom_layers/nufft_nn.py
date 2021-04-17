
from custom_layers.decoder import Decoder
from custom_layers.encoder import Encoder
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Dropout, CuDNNLSTM, Input
from tensorflow.python.keras import Model
# pylint: disable=unnecessary-semicolon

#def NUFFT_NN():
def create_nufft_nn( endcoderArch, lstm_layers,decoderArch,input_shape,CUDA=False):
    '''
    NUFFT neural net constructor
    endcoder: list 
        arguments for the encoder's architecture
    lstm_layers: list of ints
        list contains number units for each stacked layer of LSTM
    input_shape: list int
        shape of the input in list. Size of of the dimension for corresponding index
    decoderArch: list
        arguments for the decoder's architecture
    CUDA: boolean
        if set to False will not use CuDNNLSTM. Therefore it should be set False when support 
        for cuda is not available

    '''
    #{
    #super(NUFFT_NN, self).__init__()
    #isLastLSTM = True
    lstm = []
    encoder = None
    numConv = 0
    if len(endcoderArch) != 0:
        numConv = len(endcoderArch[1])
        if len(endcoderArch[1]) !=0:
            encoder = Encoder(*endcoderArch)

    if CUDA :
        #{
        if len(lstm_layers)>2:
            #{
            if numConv == 0:
                lstm.append(CuDNNLSTM(lstm_layers[0], input_shape=input_shape, return_sequences=True))
            else:
                lstm.append(CuDNNLSTM(lstm_layers[0], return_sequences=True))
            del lstm_layers[0]
            lastLayer  = lstm_layers.pop()
            CuDNNLSTM(lstm_layers[0], input_shape=input_shape, return_sequences=True)
            for ix, units in enumerate(lstm_layers):
                #{
                lstm.append(CuDNNLSTM(units,  return_sequences=True))
                #}
            lstm.append(CuDNNLSTM(lastLayer, return_sequences=False))
            #}
        
        elif len(lstm_layers) == 2:
            #{
            if numConv == 0:
                lstm.append(CuDNNLSTM(lstm_layers[0], input_shape=input_shape, return_sequences=True))
            else:
                lstm.append(CuDNNLSTM(lstm_layers[0], return_sequences=True))
            lstm.append(CuDNNLSTM(lstm_layers[1], return_sequences=False))
            #}
        else:
            #{
            if numConv == 0:
                lstm.append(CuDNNLSTM(lstm_layers[0], input_shape=input_shape, return_sequences=False))
            else:
                lstm.append(CuDNNLSTM(lstm_layers[0], return_sequences=False))
            #} 
        #]
    else:
        #{
        if len(lstm_layers)>2:
            #{
            if numConv == 0:
                lstm.append(LSTM(lstm_layers[0], input_shape=input_shape, return_sequences=True))
            else:
                lstm.append(LSTM(lstm_layers[0], return_sequences=True))
            del lstm_layers[0]
            lastLayer  = lstm_layers.pop()
            for ix, units in enumerate(lstm_layers):
            #{
                lstm.append(LSTM(units,  return_sequences=True))
            #}
            lstm.append(LSTM(lastLayer, return_sequences=False))
            #}
        elif len(lstm_layers) == 2:
            #{
            if numConv == 0:
                lstm.append(LSTM(lstm_layers[0], input_shape=input_shape, return_sequences=True))
            else:
                lstm.append(LSTM(lstm_layers[0], return_sequences=True))
            lstm.append(LSTM(lstm_layers[1], return_sequences=False))  
            #}
        else:
            if numConv == 0:
                lstm.append(LSTM(lstm_layers[0], input_shape=input_shape,return_sequences=False))
            else:
                lstm.append(LSTM(lstm_layers[0],return_sequences=False))

        #}
    decoder = Decoder( *decoderArch)
    input_points = Input(shape=input_shape)
    x =input_points
    if encoder !=None:
        x = tf.expand_dims(x,-1)
        x = encoder.call(x)
        x = tf.keras.backend.squeeze(x,-1)
    for ix in range(len(lstm)):
        x = lstm[ix](x)
    x = decoder.call(x)
    nufft = Model(input_points,x)
    return nufft
 





