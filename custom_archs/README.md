## List of architectures:

Running the scrpit  of each architecture will give a model summery about the
architecutre, number of parameters, and output shapes. After exit the process
if traning the network is not intended, ctrl +z.

y_axis_len, and x_axis_len is refereing to number
of rows and colums respectivly of matrix.
They were originally the image height and width
before NUFFT, and y_axis_len*x_axis_len
will present number of samples.

Note: All of these architecutres are hard-coded for dataset
generated from 256x256 images from ../utils/data_gen.py
due to the presence of LSTM layer. Exceptions will be noted.

    # Bidirectional LSTM  Convultional Decoder

        This architecutre is implemented in : dis_pnet_lstm_decoder.py
        The best record:
        mean_squared_error: 0.0067 -val_mean_squared_error: 0.0076
        On:
        50 epochs with learning rate of 9e-4.
        The input had two different trajectories,
        cratsian and schuffled radial.
        
        This architecutre takes matrix as input.
        Input with batches looks like:
        [batch, y_axis_len, x_axis_len * 4]
    

    # PointNet Encoder LSTM Conovlutional

        This architecutre is implemented in : pnet_lstm_decoderr.py
        Decoder is modified version of PointNet[1] 
        The best record:
        mean_squared_error: 0.0218 - val_mean_squared_error: 0.0221
        30 epochs with learning rate of 9e-4.
    
        This architecutre takes tensor as input.
        Input with batches looks like:
        [batch, y_axis_len * x_axis_len, 4]

    # Distributed Training forPointNet Encoder LSTM Conovlutional

        This architecutre is implemented in: dis_pnet_lstm_decoder.py
        Not tested yet. Deos not work for large number of samples.

        This architecutre takes tensor as input.
        Input with batches looks like:
        [batch, y_axis_len * x_axis_len, 4]

    # Covolutional LSTM Convoulational Decoder
        IMPORTANT: the code is not refactored
        Two different versions that work
        on mutliple GPUs:
        with_conv_lstm.py
        Works for traning samples less than 3600.

        Best recorded results:
        loss: 0.1113 - mean_squared_error: 0.1113

        with_conv_lstm_2.py
        
        Tries to solve the problem of possbiblity 
        of loading larger number of samples.
        (Under devolpment)

    #PointNet++ Decoder LSTM Convultional Decoder.

        This archtecture is implemented in:
        pnet2_lstm_decoder.py
        A PointNet++[2] decoder, LSTM, and 
        Convultional Decoder.
        Its devolpment is halted.







## Refrences

[1] 
Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). Pointnet: Deep learning on point sets for 3d classification and segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 652-660).

[2]
Qi, C. R., Yi, L., Su, H., & Guibas, L. J. (2017). Pointnet++: Deep hierarchical feature learning on point sets in a metric space. In Advances in neural information processing systems (pp. 5099-5108).