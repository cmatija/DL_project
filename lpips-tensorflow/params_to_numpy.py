# import sys; sys.path += ['models']
import pickle
import os
from models import dist_model as dm

use_gpu = False        # Whether to use GPU
spatial = False         # Return a spatial map of perceptual distance.
                       # Optional args spatial_shape and spatial_order control output shape and resampling filter: see DistModel.initialize() for details.

## Initializing the model
model = dm.DistModel()
model_type = 'net-lin'
model_network = 'alex'

print('used network: ' + model_network + ', type: '+model_type)
# Linearly calibrated models
model.initialize(model=model_type,net=model_network,use_gpu=use_gpu,spatial=spatial)


# Low-level metrics
# model.initialize(model='l2',colorspace='Lab')
# model.initialize(model='ssim',colorspace='RGB')
print('Model [%s] initialized'%model.name())

params_numpy = model.export_params_to_numpy()
model_path = '/home/cmatija/code/python/DL_project_github/models/'
model_name = ('alexnet' if model_network=='alex' else model_network) + '_' + model_type
pickle_out = open(model_path+model_name+'.pickle', "wb+")
pickle.dump(params_numpy, pickle_out)
pickle_out.close()