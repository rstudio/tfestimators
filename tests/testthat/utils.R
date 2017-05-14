# Util Functions

library(tensorflow)

mtcars_regression_specs <- function() {
  dnn_feature_columns <- feature_columns(mtcars, "drat")
  linear_feature_columns <- feature_columns(mtcars, "cyl")
  constructed_input_fn <- input_fn.default(mtcars, response = "mpg", features = c("drat", "cyl"))
  list(dnn_feature_columns = dnn_feature_columns,
       linear_feature_columns = linear_feature_columns,
       input_fn = constructed_input_fn)
}

mtcars_regression_specs_numpy_input_fn <- function() {
  dnn_feature_columns <- feature_columns(mtcars, "drat")
  linear_feature_columns <- feature_columns(mtcars, "cyl")
  constructed_input_fn <- input_fn(mtcars, response = "mpg", features = c("drat", "cyl"))
  list(dnn_feature_columns = dnn_feature_columns,
       linear_feature_columns = linear_feature_columns,
       input_fn = constructed_input_fn)
}

mtcars_classification_specs <- function() {
  mtcars$vs <- as.factor(mtcars$vs)
  dnn_feature_columns <- feature_columns(mtcars, "drat")
  linear_feature_columns <- feature_columns(mtcars, "cyl")
  constructed_input_fn <- input_fn.default(mtcars, response = "vs", features = c("drat", "cyl"))
  list(dnn_feature_columns = dnn_feature_columns,
       linear_feature_columns = linear_feature_columns,
       input_fn = constructed_input_fn)
}

mtcars_classification_specs_numpy_input_fn <- function() {
  mtcars$vs <- as.factor(mtcars$vs)
  dnn_feature_columns <- feature_columns(mtcars, "drat")
  linear_feature_columns <- feature_columns(mtcars, "cyl")
  constructed_input_fn <- input_fn(mtcars, response = "vs", features = c("drat", "cyl"))
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


get_non_batched_sin_input_fn <- function(sequence_length, increment, seed = NULL) {
    function(features_as_named_list) {
      function() {
        start <- random_ops$random_uniform(
          list(), minval = 0, maxval = pi * 2.0,
          dtype = tf$python$framework$dtypes$float32, seed = seed)
        sin_curves <- math_ops$sin(
          math_ops$linspace(
            start, (sequence_length - 1) * increment,
            as.integer(sequence_length + 1)))
        inputs <- array_ops$slice(sin_curves,
                                  np$array(list(0), dtype = np$int64),
                                  np$array(list(sequence_length), dtype = np$int64))
        labels <- array_ops$slice(sin_curves,
                                  np$array(list(1), dtype = np$int64),
                                  np$array(list(sequence_length), dtype = np$int64))
        tuple(list(inputs = inputs), labels)
      }
    }
}

get_batched_sin_input_fn <- function(batch_size, sequence_length, increment, seed = NULL) {
    function(features_as_named_list) {
      function() {
        starts <- random_ops$random_uniform(
          list(batch_size), minval = 0, maxval = pi * 2.0,
          dtype = tf$python$framework$dtypes$float32, seed = seed)
        sin_curves <- functional_ops$map_fn(
          function(x){
            math_ops$sin(
              math_ops$linspace(
                array_ops$reshape(x[1], list()),
                (sequence_length - 1) * increment,
                as.integer(sequence_length + 1)))
          },
          tuple(starts),
          dtype = tf$python$framework$dtypes$float32
        )
        inputs <- array_ops$expand_dims(
          array_ops$slice(
            sin_curves,
            np$array(list(0, 0), dtype = np$int64),
            np$array(list(batch_size, sequence_length), dtype = np$int64)),
          2L
        )
        labels <- array_ops$slice(sin_curves,
                                  np$array(list(0, 1), dtype = np$int64),
                                  np$array(list(batch_size, sequence_length), dtype = np$int64))
        tuple(list(inputs = inputs), labels)
      }
  }
}

fake_sequence_input_fn <- function() {
  function(unused) input_fn(
    x = list(
      features = list(
        list(list(1), list(2), list(3)),
        list(list(4), list(5), list(6))),
      response = list(
        list(1, 2, 3), list(4, 5, 6))),
    features = c("features"),
    response = "response")
}
