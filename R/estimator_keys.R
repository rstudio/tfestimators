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

print.tensorflow.python.estimator.canned.prediction_keys.PredictionKeys <- function(object) {
  cat(paste0("Available predictions keys: ", paste(names(prediction_keys()), collapse = ", ")))
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

print.tensorflow.python.estimator.canned.metric_keys.MetricKeys <- function(object) {
  cat(paste0("Available metric keys: ", paste(names(metric_keys()), collapse = ", ")))
}

#' Canonical Mode Modes
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

print.tensorflow.python.estimator.model_fn.ModeKeys <- function(object) {
  cat(paste0("Available mode keys: ", paste(names(mode_keys()), collapse = ", ")))
}


