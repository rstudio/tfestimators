

# Take a model_fn and wrap it in a python function which has the correct
# signature for model_fn. This is necessary because TF validates the signature
# of model_fn and the default Python function produced by reticulate for an
# R function has variadic arguments so fails validation.
def as_model_fn(f):
  def model_function(features, labels, mode):
    return f(features, labels, mode)
  return model_function
      
