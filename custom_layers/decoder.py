
import math
import tensorflow as tf

from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Conv2D, Conv2DTranspose, Dropout, Reshape,UpSampling2D

class Decoder():

    def __init__(self, dense_layers,conv_layers,input_shape,strides= {},kernel_sizes={},upsamplingLayers ={},
        paddings ={},isFlatten = False, isBatchNorm = (False,False),_dropout = 0.0,reg=0.0 ):
   
        '''Decoder Constructor
        dense_layers: list of ints
            The sizes of the hidden dense layers of the decoder.
        conv_layers: list of ints
            the sizes of the conv layers of the decoder.
        Input_shape:  int
            shape of the flattened input, to be converted into 2D tensor. 
        kernel_sizes: dictionary of ints
            dictionary of Indcies after which conv layer and kernel' sizes, the indcies that are not provided
            will take the default (3,3) kernel
        upsamplingLayers: dictionary of ints
            dictionary of Indcies after which conv layer and windows' sizes
        isFlatten: bool
            flattening the output of last conv layer. It is automaticall flattened if the dense layers are more than zero
        isBatchnNorm: boolean tuple
            the first indicates whether to include batch normalization for the hiiden dense layers,
            the second indicates whether to include batch normalization for the hiiden conv layers.
            by defualt everything is set to zero
        _dropout: float
            if it is set to flaot bigger than zero, then it applies the dropout according the input
            by default it is set to zero
        reg: float
            Regularization constant
        '''
        #super(Decoder, self).__init__()

        self.denseLayers = []
        self.batchNormLayers = []
        self.droputLayers = []
        self.inputShape = input_shape
        self.convLayers = []
        self.strides = strides
        self.upSamplingIndex = upsamplingLayers
        self.kernelSizes = kernel_sizes
        self.batchNormLayersConv = []
        self.droputLayersConv = []
        self.upSampLayers = []
        (self.denseBn,self.convBn)=isBatchNorm
        self.dropout = _dropout
        self.flatten = isFlatten
        self.flatten_layer = tf.keras.layers.Flatten()
        self.reshaperLayer = Reshape([int(math.sqrt(input_shape)),int(math.sqrt(input_shape)),1])
        if len(conv_layers)>0:
        

            kernel = kernel_sizes.get(0)
            if kernel is None:
                kernel = (3,3)
            stride = self.strides.get(0)
            if stride is None:
                stride = (1,1)
            padding = 'valid'
            if paddings.get(0) is not None:
                padding ='same'
            
            self.convLayers.append(Conv2DTranspose(conv_layers[0], kernel,
                                    kernel_regularizer=tf.keras.regularizers.l2(reg),
                                    padding=padding,strides=stride))
            padding = 'valid'
                
                #check whether we neen to include batch normaliztion for the hidden layers
            if self.convBn:
                self.batchNormLayersConv.append(tf.keras.layers.BatchNormalization())
            if self.dropout >0:
                self.droputLayersConv.append(Dropout(self.dropout))
            if self.upSamplingIndex.get(0) is not None :    
                self.upSampLayers.append(UpSampling2D(self.upSamplingIndex.get(0)))
            

            del conv_layers[0]
            #for each hidden conv layer
            print(len(conv_layers))
            for ix, hidden_size in enumerate(conv_layers):
                
                i = ix +1
                kernel = self.kernelSizes.get(i)
                if kernel is None:
                    kernel = (3,3)

                stride = self.strides.get(i)
                if stride is None:
                    stride = (1,1)

                if paddings.get(i) is not None:
                    padding ='same'
                self.convLayers.append(Conv2DTranspose(hidden_size,kernel,
                                        kernel_regularizer=tf.keras.regularizers.l2(reg),
                                        padding=padding,strides=stride))
                padding = 'valid'
                    #check whether we neen to include batch normaliztion for the hidden layers
                if ix is not len(conv_layers) -1:
                    if self.convBn:
                        self.batchNormLayersConv.append(tf.keras.layers.BatchNormalization())
                    if self.dropout >0:
                        self.droputLayersConv.append(Dropout(self.dropout))
                    if self.upSamplingIndex.get(i) is not None:
                        self.upSampLayers.append(UpSampling2D(self.upSamplingIndex.get(i)))
                
        
       #for each hidden dense layer    
        for ix, hidden_size in enumerate(dense_layers):
         
            self.denseLayers.append(Dense(units=hidden_size,
                kernel_regularizer=tf.keras.regularizers.l2(reg)))
            if ix is len(dense_layers)-1:
                if self.denseBn:#check whether we neen to include batch normaliztion for the hidden layers
                    self.batchNormLayers.append(tf.keras.layers.BatchNormalization())
                if self.dropout >0:
                    self.droputLayers.append(Dropout(self.dropout))
            


        
    

    

    def call(self, x):
    
        if(len(self.convLayers)!=0):
            x = self.reshaperLayer(x)

        #foreach hidden conv layer
        # 
        for ix in range(len(self.convLayers)-1):
         
            x = self.convLayers[ix](x)
            #check whether we neen to include batch normaliztion for the hidden layers
            
            if self.convBn:
                x = self.batchNormLayersConv[ix](x)

            x = tf.nn.relu(x)

            if self.dropout >0:
                x = self.droputLayersConv[ix](x)
            if self.upSamplingIndex.get(ix) is not None:
                x = self.upSampLayers[ix](x)
        
        if len(self.convLayers) !=0:
            x = self.convLayers[len(self.convLayers)-1](x)
            x = tf.keras.backend.squeeze(x,-1) 
            
        if self.flatten or len(self.denseLayers) > 0:
            x =self.flatten_layer(x)
            
        for ix in range(len(self.denseLayers)-1):
         
            x = self.denseLayers[ix](x)
            #check whether we neen to include batch normaliztion for the hidden layers
            if self.denseBn:
                x = self.batchNormLayers[ix](x)
            x = tf.nn.relu(x)
            if self.dropout >0:
                x= self.droputLayers[ix](x)
        
        if(len(self.denseLayers)>0):
            x = self.denseLayers[len(self.denseLayers)-1](x)
            #check whether we neen to include batch normaliztion for the hidden layers



        return  x
    

