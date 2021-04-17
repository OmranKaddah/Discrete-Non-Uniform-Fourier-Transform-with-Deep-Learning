## List of architectures:
All of these scrptis depend on main.pu for
describing the architecture and generating a model.
Number of layers and parameter of each can be infered from the
shell file. Refer to the main.py arguments.

Running the scrpit  of each architecture will give a model summery about the
architecutre, number of parameters After exit the process
if traning the network is not intended, ctrl +z.

y_axis_len, and x_axis_len is refereing to number
of rows and colums respectivly of matrix.
They were originally the image height and width
before NUFFT, and y_axis_len*x_axis_len
will present number of samples.


Note: All of these architecutres are created for dataset
generated from 256x256 images from ../utils/data_gen.py
due to the presence of LSTM layer. Exceptions will be noted.

    # LSTM Fully-Connected decoder

        This architecutre is implemented in : dense_layers_decoder.sh
        The best record:
        mean_squared_error: 0.05 -val_mean_squared_error: 0.08
        On:
        60 epochs with learning rate of 5e-4.
        The input had two different trajectories,
        cratsian and schuffled radial.
        Also was trained with many different 
        variations of leaning rate and number
        parameters and layers. No good results
        were recorded.
        
        This architecutre takes matrix as input.
        Input with batches looks like:
        [batch, y_axis_len, x_axis_len * 4]
    

    # LSTM Conovlutional decoder

        This architecutre is implemented in : lstm_conv_decoder.sh
        The best record:

        mean_squared_error: 0.0096 
        val_mean_squared_error: 0.0177
        Trained for 300 epochs on data with only
        radial trajectory.
        This architecutre takes tensor as input.
        Input with batches looks like:
        [batch, y_axis_len , x_axis_len * 4]

        On mutliple trajectories:
        trained for 100 epochs learning rate of 5e-4
            mean_squared_error: 0.0184 
            val_mean_squared_error: 0.0237

    
    # LSTM Conovlutional decoder v2

        This architecutre is implemented in : lstm_conv_decoder.sh
        The best record:

        mean_squared_error: 0.0096 
        val_mean_squared_error: 0.0177
        Trained for 300 epochs on data with 
        mutliple trajectories.
        This architecutre takes tensor as input.
        Input with batches looks like:
        [batch, y_axis_len ,x_axis_len * 4]

    # Conovlutional Encoder LSTM Conovlutional Decoder

        This architecutre is implemented in: encoder_lstm_decoder.py
        Not tested yet. Deos not work for large number of samples.
        The best record:
        mean_squared_error: 0.125
        val_mean_squared_error: 0.183
        Trained for 60 epochs on schuffled 
        radial and cartsian trajctory
        This architecutre takes tensor as input.
        Input with batches looks like:
        [batch, y_axis_len ,x_axis_len * 4]


