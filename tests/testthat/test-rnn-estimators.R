context("Testing rnn estimators")

source("utils.R")

test_that("state_saving_rnn_estimator works on sin sequence data", {
  np <- import("numpy", convert = FALSE)

  batch_size <- 8L
  num_unroll <- 5L
  sequence_length <- 64L
  train_steps <- 250L
  eval_steps <- 20L
  num_rnn_layers <- 1L
  num_units <- rep(4L, num_rnn_layers)
  learning_rate <- 0.3
  loss_threshold <- 0.035
  dropout_keep_probabilities <- rep(0.9, num_rnn_layers + 1)
  
  seq_columns <- list(
    tf$contrib$layers$feature_column$real_valued_column('inputs', dimension = 1L)
  )
  
  rnn_sequence_estimator <- state_saving_rnn_estimator(
    contrib_estimators_lib$constants$ProblemType$LINEAR_REGRESSION,
    num_units = num_units,
    cell_type = 'lstm',
    num_unroll = num_unroll,
    batch_size = batch_size,
    sequence_feature_columns = seq_columns,
    learning_rate = learning_rate,
    dropout_keep_probabilities = dropout_keep_probabilities,
    queue_capacity = 2 * batch_size,
    seed = 1234
  )
  
  get_sin_input_fn <- function(sequence_length, increment, seed = NULL) {
    list(
      input_fn = function() {
        start <- tf$python$ops$random_ops$random_uniform(
          tuple(), minval = 0, maxval = pi * 2.0,
          dtype = tf$python$framework$dtypes$float32, seed = seed)
        sin_curves <- tf$python$ops$math_ops$sin(
          tf$python$ops$math_ops$linspace(
            start, (sequence_length - 1) * increment,
            as.integer(sequence_length + 1)))
        inputs <- tf$python$ops$array_ops$slice(sin_curves,
                                                np$array(list(0), dtype = np$int64),
                                                np$array(list(sequence_length), dtype = np$int64))
        labels <- tf$python$ops$array_ops$slice(sin_curves,
                                                np$array(list(1), dtype = np$int64),
                                                np$array(list(sequence_length), dtype = np$int64))
        return(tuple(list(inputs = inputs), labels))},
      features_as_named_list = TRUE)
  }
  
  train_input_fn <- get_sin_input_fn(sequence_length, pi / 32, seed = 1234)
  eval_input_fn <- get_sin_input_fn(sequence_length, pi / 32, seed = 4321)
  
  fit(rnn_sequence_estimator, input_fn = train_input_fn, steps = train_steps)
  
  loss <- rnn_sequence_estimator$estimator$evaluate(input_fn = eval_input_fn$input_fn, steps = eval_steps)$loss
  expect_lte(loss, 0.05)
})
