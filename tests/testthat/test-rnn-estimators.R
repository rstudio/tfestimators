context("Testing rnn estimators")

source("utils.R")

np <- import("numpy", convert = FALSE)
random_ops <- tf$python$ops$random_ops
math_ops <- tf$python$ops$math_ops
array_ops <- tf$python$ops$array_ops
functional_ops <- tf$python$ops$functional_ops

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
  
  get_sin_input_fn <- function(sequence_length, increment, seed = NULL) {
    list(
      input_fn = function() {
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
        return(tuple(list(inputs = inputs), labels))
      },
      features_as_named_list = TRUE)
  }
  
  train_input_fn <- get_sin_input_fn(sequence_length, pi / 32, seed = 1234)
  eval_input_fn <- get_sin_input_fn(sequence_length, pi / 32, seed = 4321)
  
  fit(rnn_sequence_estimator, input_fn = train_input_fn, steps = train_steps)

  # TODO: Add predict for tf_model with type estimator
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
  
  get_sin_input_fn <- function(batch_size, sequence_length, increment, seed = NULL) {
    list(
      input_fn = function() {
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
        return(tuple(list(inputs = inputs), labels))
      },
      features_as_named_list = TRUE)
  }
  
  train_input_fn <- get_sin_input_fn(batch_size, sequence_length, pi / 32, seed = 1234)
  eval_input_fn <- get_sin_input_fn(batch_size, sequence_length, pi / 32, seed = 4321)
  
  fit(rnn_sequence_estimator, input_fn = train_input_fn, steps = train_steps)
})

