# Util Functions

library(tensorflow)

# using custom input_fn
mtcars_regression_specs <- function() {
  dnn_feature_columns <- feature_columns(column_numeric("drat"))
  linear_feature_columns <- feature_columns(column_numeric("drat"))
  constructed_input_fn <- input_fn(mtcars, response = "mpg", features = c("drat", "cyl"), batch_size = 8L)
  list(dnn_feature_columns = dnn_feature_columns,
       linear_feature_columns = linear_feature_columns,
       input_fn = constructed_input_fn)
}

mtcars_classification_specs <- function() {
  dnn_feature_columns <- feature_columns(column_numeric("drat"))
  linear_feature_columns <- feature_columns(column_numeric("drat"))
  constructed_input_fn <- input_fn(mtcars, response = "vs", features = c("drat", "cyl"))
  list(dnn_feature_columns = dnn_feature_columns,
       linear_feature_columns = linear_feature_columns,
       input_fn = constructed_input_fn)
}

iris_classification_specs <- function() {
  dnn_feature_columns <- feature_columns(column_numeric("Sepal.Length"))
  linear_feature_columns <- feature_columns(column_numeric("Sepal.Length"))
  constructed_input_fn <- input_fn(iris, response = "Species", features = c("Sepal.Length", "Sepal.Width"))
  list(dnn_feature_columns = dnn_feature_columns,
       linear_feature_columns = linear_feature_columns,
       input_fn = constructed_input_fn)
}

simple_dummy_model_fn <- function(features, labels, mode, params, config) {
  names(features) <- NULL
  features <- tf$stack(unlist(features))
  predictions <- tf$python$framework$constant_op$constant(list(runif(1, 5.0, 7.5)))
  if (mode == "infer") {
    return(estimator_spec(predictions = predictions, mode = mode, loss = NULL, train_op = NULL))
  }
  loss <- tf$losses$mean_squared_error(labels, predictions)
  train_op <- tf$python$ops$state_ops$assign_add(tf$contrib$framework$get_global_step(), 1L)
  return(estimator_spec(predictions, loss, train_op, mode))
}

simple_custom_model_fn <- function(features, labels, mode, params, config) {
  
  # Create three fully connected layers respectively of size 10, 20, and 10 with
  # each layer having a dropout probability of 0.1.
  logits <- features %>%
    tf$contrib$layers$stack(
      tf$contrib$layers$fully_connected, c(10L, 20L, 10L),
      normalizer_fn = tf$contrib$layers$dropout,
      normalizer_params = list(keep_prob = 0.9)) %>%
    tf$contrib$layers$fully_connected(3L, activation_fn = NULL) # Compute logits (1 per class) and compute loss.
  
  predictions <- list(
    class = tf$argmax(logits, 1L),
    prob = tf$nn$softmax(logits))
  
  if (mode == "infer") {
    return(estimator_spec(mode = mode, predictions = predictions, loss = NULL, train_op = NULL))
  }
  
  labels <- tf$one_hot(labels, 3L)
  loss <- tf$losses$softmax_cross_entropy(labels, logits)
  
  # Create a tensor for training op.
  train_op <- tf$contrib$layers$optimize_loss(
    loss,
    tf$contrib$framework$get_global_step(),
    optimizer = 'Adagrad',
    learning_rate = 0.1)
  
  return(estimator_spec(mode = mode, predictions = predictions, loss = loss, train_op = train_op))
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
        tuple(list(features = inputs), labels)
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
        tuple(list(features = inputs), labels)
      }
  }
}

fake_sequence_input_fn <- function() {
  input_fn(
    object = list(
      features = list(
        list(list(1), list(2), list(3)),
        list(list(4), list(5), list(6))),
      response = list(
        list(1, 2, 3), list(4, 5, 6))),
    features = c("features"),
    response = "response")
}
