# Util Functions

mtcars_regression_specs <- function() {
  dnn_feature_columns <- feature_columns(mtcars, "drat")
  linear_feature_columns <- feature_columns(mtcars, "cyl")
  constructed_input_fn <- input_fn(mtcars, response = "mpg", features = c("drat", "cyl"))
  list(dnn_feature_columns = dnn_feature_columns,
       linear_feature_columns = linear_feature_columns,
       input_fn = constructed_input_fn)
}

mtcars_regression_specs_numpy_input_fn <- function() {
  dnn_feature_columns <- feature_columns(mtcars, "drat")
  linear_feature_columns <- feature_columns(mtcars, "cyl")
  constructed_input_fn <- numpy_input_fn(mtcars, response = "mpg", features = c("drat", "cyl"))
  list(dnn_feature_columns = dnn_feature_columns,
       linear_feature_columns = linear_feature_columns,
       input_fn = constructed_input_fn)
}

mtcars_classification_specs <- function() {
  mtcars$vs <- as.factor(mtcars$vs)
  dnn_feature_columns <- feature_columns(mtcars, "drat")
  linear_feature_columns <- feature_columns(mtcars, "cyl")
  constructed_input_fn <- input_fn(mtcars, response = "vs", features = c("drat", "cyl"))
  list(dnn_feature_columns = dnn_feature_columns,
       linear_feature_columns = linear_feature_columns,
       input_fn = constructed_input_fn)
}

mtcars_classification_specs_numpy_input_fn <- function() {
  mtcars$vs <- as.factor(mtcars$vs)
  dnn_feature_columns <- feature_columns(mtcars, "drat")
  linear_feature_columns <- feature_columns(mtcars, "cyl")
  constructed_input_fn <- numpy_input_fn(mtcars, response = "vs", features = c("drat", "cyl"))
  list(dnn_feature_columns = dnn_feature_columns,
       linear_feature_columns = linear_feature_columns,
       input_fn = constructed_input_fn)
}

simple_dummy_model_fn <- function(features, labels, mode, params, config) {
  names(features) <- NULL
  features <- tf$stack(unlist(features))
  predictions <- tf$python$framework$constant_op$constant(list(runif(1, 5.0, 7.5)))
  if(mode == "infer") {
    return(estimator_spec(predictions = predictions, mode = mode, loss = NULL, train_op = NULL))
  }
  loss <- tf$losses$mean_squared_error(labels, predictions)
  train_op <- tf$python$ops$state_ops$assign_add(tf$contrib$framework$get_global_step(), 1L)
  return(estimator_spec(predictions, loss, train_op, mode))
}

