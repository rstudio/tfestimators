# Util Functions

mtcars_regression_specs <- function() {
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

simple_custom_model_fn <- function(features, labels, mode, params, config) {
  names(features) <- NULL
  features <- tf$stack(unlist(features))
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

  if(mode == "infer"){
    return(estimator_spec(predictions = predictions, mode = mode, loss = NULL, train_op = NULL))
  }

  labels <- tf$one_hot(labels, 3L)
  loss <- tf$losses$softmax_cross_entropy(labels, logits)

  # Create a tensor for training op.
  train_op <- tf$contrib$layers$optimize_loss(
    loss,
    tf$contrib$framework$get_global_step(),
    optimizer = 'Adagrad',
    learning_rate = 0.1)

  return(estimator_spec(predictions, loss, train_op, mode))
}
