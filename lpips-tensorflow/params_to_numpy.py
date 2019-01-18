# import sys; sys.path += ['models']
import pickle
import os
import sys
sys.path.insert(0, './PerceptualSimilarity/')
from models import dist_model as dm
#OUR CODE

use_gpu = False        # Whether to use GPU
spatial = False         # Return a spatial map of perceptual distance.
                       # Optional args spatial_shape and spatial_order control output shape and resampling filter: see DistModel.initialize() for details.

## Initializing the model
model = dm.DistModel()
model_type = 'net'
model_networks = ['vgg', 'alex'] #choose alex or vgg

for model_network in model_networks:
    print('used network: ' + model_network + ', type: '+model_type)
    # Linearly calibrated models
    model.initialize(model=model_type,net=model_network,use_gpu=use_gpu,spatial=spatial)


    # Low-level metrics
    # model.initialize(model='l2',colorspace='Lab')
    # model.initialize(model='ssim',colorspace='RGB')
    print('Model [%s] initialized'%model.name())

    params_numpy = model.export_params_to_numpy()
    model_path = '../models/'
    print(os.path.dirname(os.path.realpath(__file__)))
    if not os.path.exists(model_path):
        print('now creating path')
        os.makedirs(model_path)
    model_name = ('alexnet' if model_network=='alex' else model_network) + '_' + model_type
    pickle_out = open(model_path+model_name+'.pickle', "wb+")
    print('dumping pickle')
    pickle.dump(params_numpy, pickle_out)
    pickle_out.close()