import tensorflow as tf

from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Conv2D, Conv2DTranspose, Dropout, MaxPooling2D 
# pylint: disable=unnecessary-semicolon

class Encoder():

    def __init__(self, input_shape, conv_layers,strides ={}, kernel_sizes={},poolingLayers={},
        is_batchnorm = False,_dropout = 0.0,reg=0.0 ):
   
        '''Decoder Constructor
        Input_shape: list int
            shape of the input in list. Size of of the dimension for corresponding index

        conv_layers: list of ints
            the sizes of the conv layers of the decoder.
        strides: dictionary of ints
            dictionary of the indecies of the conv layers and their strides, the layers that are
            omitted have default stride of (1,1)
        kernel_sizes: dictionary of ints
            dictionary of the indecies of the conv layers and their strides, the layers that are
            omitted have default stride of (1,1)
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
        #super(Encoder, self).__init__()


        self.convLayers = []
        self.poolLayers = []
        self.batchNormLayersConv = []
        self.droputLayersConv = []
        self.poolLayersIndecies= poolingLayers
        
        
        self.convBn=is_batchnorm
        self.dropout = _dropout
        self.flatten = tf.keras.layers.Flatten()
        self.kernelSizes = kernel_sizes
        self.strides = strides
        


        kernel = self.kernelSizes.get(0)
        if kernel is None:
            kernel = (3,3)
        stride = self.strides.get(0)
        if stride is None:
            stride = (1,1)
        self.convLayers.append(Conv2D(conv_layers[0], kernel,
            kernel_regularizer=tf.keras.regularizers.l2(reg),strides=stride,input_shape=input_shape))
        del conv_layers[0]
        if self.convBn:
            self.batchNormLayersConv.append(tf.keras.layers.BatchNormalization())
        if self.dropout >0:
            self.droputLayersConv.append(Dropout(self.dropout))
        if self.poolLayersIndecies.get(0) is not None:

            self.poolLayers.append(MaxPooling2D(self.poolLayersIndecies.get(0)))
        #for each hidden conv layer
        for ix, hidden_size in enumerate(conv_layers):
            i = ix +1
            kernel = self.kernelSizes.get(i)
            if kernel is None:
                kernel = (3,3)
            stride = self.strides.get(i)
            if stride is None:
                stride = (1,1)
            self.convLayers.append(Conv2D(hidden_size, kernel,
                kernel_regularizer=tf.keras.regularizers.l2(reg),input_shape=input_shape,strides=stride))
            
                #check whether we neen to include batch normaliztion for the hidden layers
            if self.convBn:
                self.batchNormLayersConv.append(tf.keras.layers.BatchNormalization())
            if self.dropout >0:
                self.droputLayersConv.append(Dropout(self.dropout))
            if self.poolLayersIndecies.get(i) is not None:
                self.poolLayers.append(MaxPooling2D(self.poolLayersIndecies.get(i)))

       

    

    def call(self, x):
   
        #for each hidden conv layer
        for ix in range(len(self.convLayers)):
        
            
            x = self.convLayers[ix](x)
            if self.convBn:#check whether we neen to include batch normaliztion for the hidden layers
                x = self.batchNormLayersConv[ix](x)
            x = tf.nn.relu(x)
            if self.dropout >0:
                x = self.droputLayersConv[ix](x)
            if self.poolLayersIndecies.get(ix) is not None:
                x =self.poolLayers[ix](x)
             
        if len(self.convLayers) ==0:
            x =self.flatten(x)
        
        return x
            

    