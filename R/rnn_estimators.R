#' @export
state_saving_rnn_estimator <- function(
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
  run_options = NULL)
{
  run_options <- run_options %||% run_options()
  rnn_estimator <- contrib_estimators_lib$state_saving_rnn_estimator$StateSavingRnnEstimator(
    problem_type = problem_type,
    num_unroll = num_unroll,
    batch_size = batch_size,
    sequence_feature_columns = sequence_feature_columns,
    context_feature_columns = context_feature_columns,
    num_classes = num_classes,
    num_units = num_units,
    cell_type = cell_type,
    optimizer_type = optimizer_type,
    learning_rate = learning_rate,
    predict_probabilities = predict_probabilities,
    momentum = momentum,
    gradient_clipping_norm = gradient_clipping_norm,
    dropout_keep_probabilities = dropout_keep_probabilities,
    feature_engineering_fn = feature_engineering_fn,
    num_threads = num_threads,
    queue_capacity=queue_capacity,
    seed = seed,
    model_dir = run_options$model_dir,
    config = run_options$run_config
  )
  
  tf_model(
    c("state_saving_rnn", "estimator"),
    estimator = rnn_estimator
  )
}
