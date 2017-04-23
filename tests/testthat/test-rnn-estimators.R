context("Testing rnn estimators")

source("utils.R")

test_that("state_saving_rnn_estimator works on sine sequence data", {
  batch_size <- 8L
  num_unroll <- 5L
  sequence_length <- 64L
  train_steps <- 2L
  eval_steps <- 2L
  num_rnn_layers <- 1L
  num_units <- rep(4L, num_rnn_layers)
  learning_rate <- 0.3
  loss_threshold <- 0.035
  dropout_keep_probabilities <- rep(0.9, num_rnn_layers + 1)
  
  seq_columns <- list(
    tf$contrib$layers$feature_column$real_valued_column('inputs', dimension = 1L)
  )
  
  rnn_sequence_estimator <- state_saving_rnn_estimator(
    problem_type = contrib_estimators_lib$constants$ProblemType$LINEAR_REGRESSION,
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
  
  train_input_fn <- get_non_batched_sin_input_fn(sequence_length, pi / 32, seed = 1234)
  eval_input_fn <- get_non_batched_sin_input_fn(sequence_length, pi / 32, seed = 4321)
  
  fit(rnn_sequence_estimator, input_fn = train_input_fn, steps = train_steps)
})


test_that("dynamic_rnn_estimator works on sine sequence data", {
  
  batch_size <- 8L
  sequence_length <- 64L
  train_steps <- 2L
  eval_steps <- 2L
  num_rnn_layers <- 1L
  num_units <- rep(4L, num_rnn_layers)
  learning_rate <- 0.3
  loss_threshold <- 0.035
  dropout_keep_probabilities <- rep(0.9, num_rnn_layers + 1)
  
  seq_columns <- list(
    tf$contrib$layers$feature_column$real_valued_column('inputs', dimension = num_units[1])
  )
  
  rnn_sequence_estimator <- dynamic_rnn_estimator(
    problem_type = contrib_estimators_lib$constants$ProblemType$LINEAR_REGRESSION,
    prediction = contrib_estimators_lib$rnn_common$PredictionType$MULTIPLE_VALUE,
    num_units = num_units,
    cell_type = 'lstm',
    sequence_feature_columns = seq_columns,
    learning_rate = learning_rate,
    dropout_keep_probabilities = dropout_keep_probabilities
  )
  
  train_input_fn <- get_batched_sin_input_fn(batch_size, sequence_length, pi / 32, seed = 1234)
  eval_input_fn <- get_batched_sin_input_fn(batch_size, sequence_length, pi / 32, seed = 4321)
  
  fit(rnn_sequence_estimator, input_fn = train_input_fn, steps = train_steps)
})

