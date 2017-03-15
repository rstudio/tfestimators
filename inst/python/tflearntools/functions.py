

import rpycall

# Take an R model_fn and wrap it in a python function which has the correct
# signature for model_fn. This is necessary because TF validates the signature
# of model_fn and the default Python function produced by reticulate for an
# R function has variadic arguments so fails validation.
def create_model_fn(f):
  
  def model_function(features, labels, mode, params, config):
    return rpycall.call_r_function(f, 
                                   features = features, 
                                   labels = labels, 
                                   mode = mode, 
                                   params = params, 
                                   config = config)

  return model_function
