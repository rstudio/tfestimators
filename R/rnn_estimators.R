## TODO: export once it's documented?
## #' State Saving RNN Estimator
## #' 
## #' @export
## #' @family canned estimators
## state_saving_rnn <- function(problem_type,
##                              num_unroll,
##                              batch_size,
##                              sequence_feature_columns,
##                              context_feature_columns = NULL,
##                              num_classes = NULL,
##                              num_units = NULL,
##                              cell_type = 'basic_rnn',
##                              optimizer_type = 'SGD',
##                              learning_rate = 0.1,
##                              predict_probabilities = F,
##                              momentum = NULL,
##                              gradient_clipping_norm = 5.0,
##                              dropout_keep_probabilities = NULL,
##                              feature_engineering_fn = NULL,
##                              num_threads = 3L,
##                              queue_capacity = 1000L,
##                              seed = NULL,
##                              model_dir = NULL,
##                              config = NULL)
## {
##   estimator <- py_suppress_warnings(
##     contrib_estimators_lib$state_saving_rnn_estimator$StateSavingRnnEstimator(
##       problem_type = problem_type,
##       num_unroll = as.integer(num_unroll),
##       batch_size = as.integer(batch_size),
##       sequence_feature_columns = sequence_feature_columns,
##       context_feature_columns = context_feature_columns,
##       num_classes = as_nullable_integer(num_classes),
##       num_units = as_nullable_integer(num_units),
##       cell_type = cell_type,
##       optimizer_type = optimizer_type,
##       learning_rate = learning_rate,
##       predict_probabilities = predict_probabilities,
##       momentum = momentum,
##       gradient_clipping_norm = gradient_clipping_norm,
##       dropout_keep_probabilities = dropout_keep_probabilities,
##       feature_engineering_fn = feature_engineering_fn,
##       num_threads = as.integer(num_threads),
##       queue_capacity = as.integer(queue_capacity),
##       seed = as_nullable_integer(seed),
##       model_dir = resolve_model_dir(model_dir),
##       config = config
##     )
##   )
##   
##   classes <- c("regressor", "classifier", "state_saving_rnn_estimator")
##   tf_estimator(estimator, classes)
## }


#' Dynamic RNN Estimator
#' 
#' @inheritParams estimators
#' 
#' @param problem_type whether the `Estimator` is intended for a regression or
#'   classification problem. Value must be one of `ProblemType.CLASSIFICATION`
#'   or `ProblemType.LINEAR_REGRESSION`.
#' @param prediction_type whether the `Estimator` should return a value for each
#'   step in the sequence, or just a single value for the final time step. Must
#'   be one of `PredictionType.SINGLE_VALUE` or `PredictionType.MULTIPLE_VALUE`.
#' @param sequence_feature_columns An iterable containing all the feature
#'   columns describing sequence features. All items in the iterable should be
#'   instances of classes derived from `FeatureColumn`.
#' @param context_feature_columns An iterable containing all the feature columns
#'   describing context features, i.e., features that apply across all time
#'   steps. All items in the set should be instances of classes derived from
#'   `FeatureColumn`.
#' @param num_classes the number of classes for a classification problem. Only
#'   used when `problem_type=ProblemType.CLASSIFICATION`.
#' @param num_units A list of integers indicating the number of units in the
#'   `RNNCell`s in each layer.
#' @param cell_type A subclass of `RNNCell` or one of 'basic_rnn,' 'lstm' or
#'   'gru'.
#' @param optimizer Either the name of the optimizer to be used when training
#'   the model, or a `tf$Optimizer` instance. Defaults to the SGD optimizer.
#' @param learning_rate Learning rate. This argument has no effect if
#'   `optimizer` is an instance of an `Optimizer`.
#' @param predict_probabilities A boolean indicating whether to predict
#'   probabilities for all classes. Used only if `problem_type` is
#'   `ProblemType.CLASSIFICATION`
#' @param momentum Momentum value. Only used if `optimizer_type` is 'Momentum'.
#' @param gradient_clipping_norm Parameter used for gradient clipping. If
#'   `NULL`, then no clipping is performed.
#' @param dropout_keep_probabilities a list of dropout probabilities or `NULL`.
#'   If a list is given, it must have length `len(num_units) + 1`. If `NULL`,
#'   then no dropout is applied.
#' @param feature_engineering_fn Takes features and labels which are the output
#'   of `input_fn` and returns features and labels which will be fed into
#'   `model_fn`. Please check `model_fn` for a definition of features and
#'   labels.
#' @export
#' @family canned estimators
dynamic_rnn <- function(problem_type,
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
  estimator <- contrib_estimators_lib$dynamic_rnn_estimator$DynamicRnnEstimator(
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
    model_dir = resolve_model_dir(model_dir),
    config = config
  )
  
  tf_estimator(estimator, "dynamic_rnn_estimator")
}

