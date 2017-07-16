#' State Saving RNN Estimator
#' 
#' @export
#' @family canned estimators
state_saving_rnn <- function(
  problem_type,
  num_unroll,
  batch_size,
  sequence_feature_columns,
  context_feature_columns = NULL,
  num_classes = NULL,
  num_units = NULL,
  cell_type = 'basic_rnn',
  optimizer_type = 'SGD',
  learning_rate = 0.1,
  predict_probabilities = F,
  momentum = NULL,
  gradient_clipping_norm = 5.0,
  dropout_keep_probabilities = NULL,
  feature_engineering_fn = NULL,
  num_threads = 3L,
  queue_capacity = 1000L,
  seed = NULL,
  model_dir = NULL,
  config = NULL)
{
  rnn_estimator <- py_suppress_warnings(contrib_estimators_lib$state_saving_rnn_estimator$StateSavingRnnEstimator(
    problem_type = problem_type,
    num_unroll = as.integer(num_unroll),
    batch_size = as.integer(batch_size),
    sequence_feature_columns = sequence_feature_columns,
    context_feature_columns = context_feature_columns,
    num_classes = as_nullable_integer(num_classes),
    num_units = as_nullable_integer(num_units),
    cell_type = cell_type,
    optimizer_type = optimizer_type,
    learning_rate = learning_rate,
    predict_probabilities = predict_probabilities,
    momentum = momentum,
    gradient_clipping_norm = gradient_clipping_norm,
    dropout_keep_probabilities = dropout_keep_probabilities,
    feature_engineering_fn = feature_engineering_fn,
    num_threads = as.integer(num_threads),
    queue_capacity = as.integer(queue_capacity),
    seed = as_nullable_integer(seed),
    model_dir = model_dir,
    config = config
  ))
  
  tf_model(
    c("state_saving_rnn", "estimator"),
    estimator = rnn_estimator
  )
}


#' Dynamic RNN Estimator
#' 
#' @export
#' @family canned estimators
dynamic_rnn <- function(
  problem_type,
  prediction_type,
  sequence_feature_columns,
  context_feature_columns = NULL,
  num_classes = NULL,
  num_units = NULL,
  cell_type = 'basic_rnn',
  optimizer = 'SGD',
  learning_rate = 0.1,
  predict_probabilities = F,
  momentum = NULL,
  gradient_clipping_norm = 5.0,
  dropout_keep_probabilities = NULL,
  feature_engineering_fn = NULL,
  model_dir = NULL,
  config = NULL)
{
  rnn_estimator <- contrib_estimators_lib$dynamic_rnn_estimator$DynamicRnnEstimator(
    problem_type = problem_type,
    prediction_type = prediction_type,
    sequence_feature_columns = sequence_feature_columns,
    context_feature_columns = context_feature_columns,
    num_classes = as_nullable_integer(num_classes),
    num_units = as_nullable_integer(num_units),
    cell_type = cell_type,
    optimizer = optimizer,
    learning_rate = learning_rate,
    predict_probabilities = predict_probabilities,
    momentum = momentum,
    gradient_clipping_norm = gradient_clipping_norm,
    dropout_keep_probabilities = dropout_keep_probabilities,
    feature_engineering_fn = feature_engineering_fn,
    model_dir = model_dir,
    config = config
  )
  
  tf_model(
    c("dynamic_rnn", "estimator"),
    estimator = rnn_estimator
  )
}

