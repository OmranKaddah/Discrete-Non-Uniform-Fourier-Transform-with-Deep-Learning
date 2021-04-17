# Continuous non-uniform fast Fourier transformation using Deep  Learning
Python version: >= 3.6
Tensorflow version: >=1.14
Tested on ubuntu 16.04 and 18.04
Python libs: Pillow ,and sigpy
The dataset is to be placed in a datasets folder, that does not appear here in the repository.

Some of the architecture that were generated by main.py are sotred as shell script 
in architectures folders.

Other archtectures that were not generated by this script, are stored in custom_archs,
more information on those archtectures can be found in the custom archtectures folder.



load_model.py loads a custom architecutre model that has been saved in saved_models
folder. The arguments for it can be found below in the command arguments section.
--load_model argument should be passed.

An excpetiion is an archtecture that uses PointNet++ as encoder, requires Pytorch,
is placed in a seprate folder under PNet++.

Note: to make things simpler, /logs, /saved_predictions, /training_checkpoints
should be emptied for every new experiments.

Commands are to be run at the root of this repository .
# # Tensorboard
For tensorboard run:

tensorboard --logdir='logs/'


# # Dataset Generation
	Any kind of images can be used for this 
	project, they should all have the same size/shape.
	The project was test on 256x256 images.

	Run the following commands:
		sh utils/data_dir_creator.sh
	
	then place train and validation in datasets/images/train
	test set in datasets/images/test

	Then run:
		sh utils/indexer.sh
	
	In the end folder structure should be:
	dataset/
		images/
			train/
			test/
		input/
			train/
			test/
		ground_truth/
			train/
			input/

	Note: train folders contain both training and validation data.
	All the contents of the should be numbered from 1 to num of samples.

	Run:

	 python utils/data_gen --num INSERT NUMBER OF SAMPLES  
	 python utils/data_gen --num INSERT NUMBER OF TEST SAMPLES -- cat test

	Or for ground truth that is only regridded( for a network that does not do IFFT)
	 python utils/data_gen.py --num INSERT NUMBER OF SAMPLES  --FTT_input True
	 python utils/data_gen.py --num INSERT NUMBER OF TEST SAMPLES -- cat test --FTT_input True

	 Note if the images are not of 256x256 then pass --image_hight # --image_width #
	 Warning: This function loads the whole dataset into the RAM.
	 Therefore, a RAM > 50 GB is recommended. Otherwise modeify traning loop 
	 accordingly.

	 The data generated is inputs and ground truth seprated in folders.
	 The inputs are of shape [images hight, image width * 4]
	 The ground truths are of shape [images hight *2 , image width ]

	 Regarding the dataloader, please check the arguments, they explain everything.
# Common commands arguments in the whole project
	An bit of an exception is for what is in PNet++.
	The default of all commands can vbe found in
	utils/args_parsing
	
	Optimization:
		--batch_size
			pass one int
			
		--epochs
			# of Epochs, one int.

		--learning_rate
			pass one float.

		--momentum
			pass one float.

		--num_samples
			pass number of samples to be loaded
		
		--decay
			pass float
			learning rate decay
		--loss
			pass MSE or MAE

	Data loading Arguments: 
		--is_flat_channel bool
			If set to true,the input element, 
			real, imaginary, x_coor,and y_coor 
			into flat channel.(vector with channels)
			The input will have a shape of 
			[batch, y_axis_len*x_axis_len, 4]
			Note:
			y_axis_len * x_axis_len = num of points.

		--is_flatten_gt bool
			ground truth will have 
			only one dimension.
			This is used when the
			last layer in the network
			is a fully connected layer.

		--validation_ratio
			pass the ratio of the validation data 
			of the loaded samples. 

		--num_test_samples
			pass one int of the number of traning samples.
	Model status arguments:
		--save_weights  bool --every # 
			bool will save checkpoints of the model's wieghts every # epoch.

		--load_weights bool
			pass the True to load the wieghts of the last checkpoint.

		--is_train bool
			if passed True, the the model will skip training phase.
			Useful in case of evaluation.
		--is_eval bool
			if passed True, the the model will skip evaluation phase.
			Useful in case of evaluation.

		--save_train_prediction
			pass the number of prediction of samples from training
			set to be saved.
		
		--save_test_prediction
			pass the number of prediction of samples from test
			set to be saved.
		--save_inputs
			saves the inputs used for predicitons
		--verbose
			pass bool
			if set to ture it will show the logs for every batch

		--name_model str
			pass the name of the model to be saved.
			If a name is passed the mdoel is
			going to be saved in saved_models folder.
			Otherwise the model is not going to be saved.
			Works for load_model.py and custom_archs.
		--recompile bool
			if passed true, it will recomiple the model
			making it possible to change the learning rate,
			optimizer, and decay. However, the status
			of the previous optimzer will be lost
	Only for load_model.py
		--load_model str
			pass the name of the model to be loaded.
			The models are saved. 
			This command can be executed with
			load_model.py
			IMPORTANT: THis argument should be passed
			for load_model.py

# visualization of results

	running display.py --num #
	--cat str --image_height # --image_weidth

	specify the number of prediction that are in 
	saved_predictions folder to be visualized.

	If the test predcitions are to be visualized
	pass --cat test.
	If the model trained on data generated 
	from image of different shape than 256x256.
	Then pass the --image_height --image_width

	After running this script, you should
	see three subplots, one for trajectory,
	prediction, and ground truth.


	
# # Instruction on main.py
The general architecture in main.py looks like:

	           		     	       	
conv downsampling layers  ===>  lstm layers  ==>  conv upsampling layers ===>  dense layers 

Any of these parts can be skipped.

To run the model:
python3 main.py 

or with passed arguments e.g:
python3 main.py --input_yx_size 1280 720  ----is_CuDNNLSTM True --dense_layers 2000 1000

This command will intialize the model with the default settings specidfied in main.py. 
You can find the defualt settings with #default on the line of the a specific parameter of the model.

Note: This model's weights can be saved. However, in this version of Tensorflow, the entire model 
can not be saved.
		

# List of arguments for main.py:
	# Encoder Architecture

		--conv_layers
			pass int numbers of the nubmer of units in a convolutional 
			layer the number of passed integers also determine the  
			number of conv layers.
			e.g --conv_layers 64 32.
			model will have two conv layers with 64 units for the 
			first and 32 for the seond.
		--kernel_sizes_encoder
			pass for every specified layer three ints:
			# the layer index(startinng from zero)
			# the kernel width
			# the kernel height
			the layers that are not specified will have (3,3)
			kernel by default.
			Example: 
			--kernel_sizes_encoder 2 5 5 6 2 2
			layer indexed 2 will have (5,5) kernel
			layer indexed 6 will have (2,2) kernel
			other layers will have by default (3,3).

		--strides_encoder
			pass for every specified layer three ints:
			# the layer index(startinng from zero)
			# the stride width
			# the stride height
			the layers that are not specified will have (1,1)
			stride by default.
			Example: 
			--strides_encoder 3 2 3 6 4 2
			layer indexed 3 will have (2,3) stride
			layer indexed 6 will have (4,2) stride
			other layers will have by default (1,1).	


		--pooling_layers
			max pool layer
			pass for every specified layer three ints:
			# the layer index(startinng from zero)
			# the window's width
			# the window's height
			the layers that are not specified will have (1,1)
			stride by default.
			Example: 
			--pooling_layers 2 2 2 6 4 2
			after layer 3 will have (2,2) maxpool layer
			after layer 6 will have (4,2) maxpool layer
			other layers will have by default (1,1)

		--input_yx_size
			pass two int numbers of the dimension of the input.
		

		--lstm_layers
			pass int numbers of the nubmer of units in a lstm cell
			the number of passed integers also determine the number 
			of lstm layers.
			e.g --lstm_layers 1024 256.
			model will have two lstm layers with 1024 units for the 
			first and 256 for the seond.
			IMPORTANT NOTE: the last hidden units should have integer
			that has an integer square root.

	# Decoder Architecture

		--dense_layers
			pass int numbers of the nubmer of units in a layer
			the number of passed integers also determine the number 
			of lstm layers.
			e.g --dense_layers 1024 512.
			model will have two layers with 1024 units for the 
			first and 512 for the seond.

		--trans_conv_layers
			pass int numbers of the nubmer of units in a conv layer
			the number of passed integers also determine the number 
			of lstm layers.
			e.g --trans_conv_layers 64 128.
			model will have two transpose conv layers with 64 units 
			for the first and 128 for the seond.	

		--kernel_sizes_decoder
			pass for every specified layer three ints:
			# the layer index(startinng from zero)
			# the kernel width
			# the kernel height
			the layers that are not specified will have (3,3)
			kernel by default.
			Example: 
			--kernel_sizes_decoder 2 5 5 6 2 2
			layer indexed 2 will have (5,5) kernel
			layer indexed 6 will have (2,2) kernel
			other layers will have by default (3,3).

		--strides_decoder
			pass for every specified layer three ints:
			# the layer index(startinng from zero)
			# the stride width
			# the stride height
			the layers that are not specified will have (1,1)
			stride by default.
			Example: 
			--strides_decoder 3 2 3 6 4 2
			layer indexed 3 will have (2,3) stride
			layer indexed 6 will have (4,2) stride
			other layers will have by default (1,1).	


		--upsampling_layers
			max pool layer
			pass for every specified layer three ints:
			# the layer index(startinng from zero)
			# the window's width
			# the window's height
			the layers that are not specified will have (1,1)
			stride by default.
			Example: 
			--upsampling_layers 2 2 2 6 4 2
			after layer 3 will have (2,2) upsampling layer
			after layer 6 will have (4,2) upsampling layer
			other layers will have by default (1,1).

		--is_batchnorm_conv
			pass bool for enabling batch  conv layers.
		
		--is_batchnorm_dense
			pass bool for enabling batch  dense layers.

		--dropout
			pass one flaot.

		--reg
			pass one flaot.
			adds regularization in all of the layers

		--is_CuDNNLSTM
			pass one bool
			if support for CuDNNLSTM is available.