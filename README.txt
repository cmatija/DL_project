DATA PREPARATION:

	1. download KODAK dataset (http://r0k.us/graphics/kodak/) to folder [kodak]
	2. go to https://github.com/fab-jul/imgcomp-cvpr and follow their data preparation steps, which collects images from imagenet into tf-records. 
	Using their steps on a more or less decent machine can finish this overnight. Let's assume the folder containing the data is called [data]


ENVIRONMENT PREPARATION:
	suggested procedure: use conda
	
	0. Collect project into [project_folder], containting [project_folder]/imgcomp-cvpr and [project_folder]/lpips-tensorflow
	
	1. create conda environment using python==3.6.7, and activate that environment

	2. go to [project_folder], run pip install -r requirements.txt

	3. go to [project_folder]/lpips_tensorflow, run python params_to_numpy.py
		--> this exports weight parameters used in LPIPS for AlexNet and VGG to [project_folder]/models


RUNNING INSTRUCTIONS

	4. At this point, you can test whether the setup works: 
		- Download the checkpoints from https://github.com/fab-jul/imgcomp-cvpr to a folder [ckpts]
		- go to [project_folder]/imgcomp-cvpr/code
		- run: python val.py [ckpts] use_all [kodak]/\*.png --save_ours
			--> goes through all the configs in [ckpts] and evaluates the LAST stored checkpoint for each config (generated during training), 
			stores compressed images within [ckpts] and outputs evaluation metrics for each image.
			--> while running that, LPIPS-code downloads a .pb file into [project_folder]/models which contains weights used by lpips_tensorflow directly.
				The networ that it downloads is specified in val.py by means of the variable 'network' (can be either 'vgg' or 'alexnet'
			(used to compare distances as computed by our network implementations and the original LPIPS ones)
			
	5. If step 4 worked, training will also work as follows:
	
		- run export RECORDS_ROOT=[data]/records in terminal
		- choose a directory to store log-data of the training procedure. Hereinafter, we denote it as [logdir]
		- run: python train.py [project_folder]/imgcomp-cvpr/code/ae_configs/cvpr_arch_param_b_5/hi_alexnet_vgg_half_250 [project_folder]/imgcomp-cvpr/code/pc_configs/cvpr/res_shallow --dataset_train imgnet_train --dataset_test imgnet_test --log_dir_root [logdir
		
		--> this uses the example config file hi_alexnet_vgg_half_250, corresponding to the config denoted as alexnet\_vgg_{250, high} in our report.
		
		
CONFIG FILE EXPLANATION:
	
	As mentioned above, the config files for the different experiments we ran are located in [project_folder]/imgcomp-cvpr/code/ae_configs/cvpr_arch_param_b_5/.
	
	-low, med, hi are baseline configs used by Mentzer et al. We didn't use those, instead, we directly worked with the checkpoints they provided (see step 4.1)
	- e.g. hi_alexnet, hi_alexnet_500, hi_alexnet250 correspond to configurations using AlexNet for the loss, using weights (1000, 500, 250), respectively
		(the base value used in hi_alexnet is stored in  [project_folder]/imgcomp-cvpr/code/ae_configs/cvpr_arch_param_b_5/base. look for K_perceptual
		
	- for mixed configs (e.g. med_alexnet_vgg_125), the 125 corresponds to a c_loss of 2*125=250 (i.e. computing loss using alexnet and vgg using c_loss=250, and then taking the mean of those 2 values)
		--> in the implementation, we just add the two losses (from alexnet and vgg), but use a weight of c_loss/2, hence the weight in the config file name is always half of the c_loss config
		as described in the paper
		
	- note: all config files we used (low_*, med_*, hi_*) were generated using the baseline files low, med, hi
