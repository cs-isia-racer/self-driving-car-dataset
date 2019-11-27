import numpy as np

from sklearn.svm import SVR

def svr_from_config(config):
    m = SVR()
    m.set_params(**config['params'])
    
    for attr, v in config['attributes'].items():
        dtype = config['attributes_types'].get(attr, 'float64')
        if isinstance(v, list):
            v = np.array(v, dtype=dtype)
        m.__setattr__(attr, v)
        
    return m

def extract_svr_config(model):
    config = {
        "params": model.get_params(),
        "attributes": {},
        "attributes_types": {}
    }
    
    attrs = [
        'support_', 'support_vectors_', '_sparse', 'shape_fit_', 
        'n_support_', '_dual_coef_', '_intercept_',
        'probA_', 'probB_', '_gamma'
    ]
    
    for attr in attrs:
        v = model.__getattribute__(attr)
        if isinstance(v, np.ndarray):
            config['attributes_types'][attr] = v.dtype.name
            v = v.tolist()
        config['attributes'][attr] = v
        
    return config