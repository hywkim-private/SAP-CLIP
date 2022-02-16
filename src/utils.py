import numpy as np

#temporary code to get test the hard-coded version of point prediction network
def update_recursive(dict1, dict2):
    '''
    #the following code is copied from https://github.com/autonomousvision/shape_as_points/
    Update two config dictionaries recursively.
    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used
    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

#given number of points to sample, return uniformly-sampled coordinates
def sample_coordinates_uniform(num_samples):
    '''
    Args:
        num_samples: number of points to sample
    '''
    ls = np.linspace(-1,1,num_samples)
    X,Y,Z = np.meshgrid(ls, ls, ls)
    pt = np.stack((X,Y,Z), axis=-1)
    pt = np.reshape(pt, (-1, 3))
    return pt
    
    

#custom function for printing parameters of a model
def print_params(model):
    for param_model in model.state_dict():
        print(param_model, "\t", model.state_dict()[param_model].size())