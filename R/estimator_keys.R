available_keys <- function(keys) {
  unlist(lapply(names(keys), function(x) keys[[x]]))
}

#' Canonical Model Prediction Keys
#' 
#' The canonical set of keys used for models and estimators that provide
#' different types of predicted values through their `predict()` method.
#' 
#' @export
#' @examples 
#' keys <- prediction_keys()
#' 
#' # Get the available keys
#' keys
#' 
#' # Key for retrieving probabilities from prediction values
#' keys$PROBABILITIES
#' @family estimator keys
prediction_keys <- function() {
  canned_estimator_lib$prediction_keys$PredictionKeys()
}

#' @export
print.tensorflow.python.estimator.canned.prediction_keys.PredictionKeys <- function(x, ...) {
  cat(paste0(
    "Available predictions keys: ",
    paste(
      available_keys(prediction_keys()),
      collapse = ", ")))
}


#' Canonical Metric Keys
#' 
#' The canonical set of keys that can be used to access metrics from canned
#' estimators.
#' 
#' @examples 
#' metrics <- metric_keys()
#' 
#' # Get the available keys
#' metrics
#' 
#' metrics$ACCURACY
#' 
#' @export
#' @family estimator keys
metric_keys <- function() {
  canned_estimator_lib$metric_keys$MetricKeys()
}

#' @export
print.tensorflow.python.estimator.canned.metric_keys.MetricKeys <- function(x, ...) {
  cat(paste0("Available metric keys: ",
             paste(available_keys(metric_keys()), collapse = ", ")))
}


#' Canonical Mode Keys
#' 
#' The names for different possible modes for an estimator. The following
#' standard keys are defined:
#' 
#' \tabular{ll}{
#' `TRAIN`   \tab Training mode.               \cr
#' `EVAL`    \tab Evaluation mode.             \cr
#' `PREDICT` \tab Prediction / inference mode. \cr
#' }
#' 
#' @examples 
#' modes <- mode_keys()
#' modes$TRAIN
#' 
#' @export
#' @family estimator keys
mode_keys <- function() {
  tf$estimator$ModeKeys()
}

#' @export
print.tensorflow.python.estimator.model_fn.ModeKeys <- function(x, ...) {
  cat(paste0("Available mode keys: ", paste(available_keys(mode_keys()), collapse = ", ")))
}


#' Standard Names to Use for Graph Collections
#' 
#' The standard library uses various well-known names to collect and retrieve 
#' values associated with a graph.
#' 
#' For example, the `tf$Optimizer` subclasses default to optimizing the 
#' variables collected under`graph_keys()$TRAINABLE_VARIABLES` if `NULL` is 
#' specified, but it is also possible to pass an explicit list of variables.
#' 
#' The following standard keys are defined:
#' 
#' * `GLOBAL_VARIABLES`: the default collection of `Variable` objects, shared 
#' across distributed environment (model variables are subset of these). See 
#' `tf$global_variables` for more details. Commonly, all `TRAINABLE_VARIABLES` 
#' variables will be in `MODEL_VARIABLES`, and all `MODEL_VARIABLES` variables 
#' will be in `GLOBAL_VARIABLES`.
#' 
#' * `LOCAL_VARIABLES`: the subset of `Variable` objects that are local to each 
#' machine. Usually used for temporarily variables, like counters. Note: use 
#' `tf$contrib$framework$local_variable` to add to this collection.
#' 
#' * `MODEL_VARIABLES`: the subset of `Variable` objects that are used in the 
#' model for inference (feed forward). Note: use 
#' `tf$contrib$framework$model_variable` to add to this collection.
#' 
#' * `TRAINABLE_VARIABLES`: the subset of `Variable` objects that will be
#' trained by an optimizer. See `tf$trainable_variables` for more details.
#' 
#' * `SUMMARIES`: the summary `Tensor` objects that have been created in the 
#' graph. See `tf$summary$merge_all` for more details.
#' 
#' * `QUEUE_RUNNERS`: the `QueueRunner` objects that are used to produce input
#' for a computation. See `tf$train$start_queue_runners` for more details.
#' 
#' * `MOVING_AVERAGE_VARIABLES`: the subset of `Variable` objects that will also
#' keep moving averages. See `tf$moving_average_variables` for more details.
#' 
#' * `REGULARIZATION_LOSSES`: regularization losses collected during graph 
#' construction. The following standard keys are defined, but their 
#' collections are **not** automatically populated as many of the others are:
#'   * `WEIGHTS` 
#'   * `BIASES` 
#'   * `ACTIVATIONS`
#' 
#' @examples 
#' graph_keys()
#' graph_keys()$LOSSES
#' 
#' @export
#' @family utility functions
graph_keys <- function() {
  tf$python$framework$ops$GraphKeys()
}

#' @export
print.tensorflow.python.framework.ops.GraphKeys <- function(x, ...) {
  cat(paste0("Available graph keys: ", paste(available_keys(graph_keys()), collapse = ", ")))
}

