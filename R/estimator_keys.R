#' Enum for canonical model prediction keys.
#' 
#' Used by models that predict values, such as regressor models.
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

print.tensorflow.python.estimator.canned.prediction_keys.PredictionKeys <- function(object) {
  cat(paste0("Available predictions keys: ", paste(names(mode_keys()), collapse = ", ")))
}

#' Enum for metric keys. 
#' 
#' Used by retrieving available metrics from canned estimators.
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

print.tensorflow.python.estimator.canned.metric_keys.MetricKeys <- function(object) {
  cat(paste0("Available metric keys: ", paste(names(mode_keys()), collapse = ", ")))
}

#' Standard names for model modes.
#' 
#' The following standard keys are defined: 
#' 
#' * `TRAIN`: training mode.
#' * `EVAL`: evaluation mode.
#' * `PREDICT`: inference mode.
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

print.tensorflow.python.estimator.model_fn.ModeKeys <- function(object) {
  cat(paste0("Available mode keys: ", paste(names(mode_keys()), collapse = ", ")))
}


