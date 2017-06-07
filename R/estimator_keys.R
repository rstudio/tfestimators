#' Enum for canonical model prediction keys.
#' 
#' Used by models that predict values, such as regressor models.
#' 
#' @export
#' @examples 
#' keys <- prediction_keys()
#' 
#' # Get the available keys
#' names(keys)
#' 
#' # Key for retrieving probabilities from prediction values
#' keys$PROBABILITIES
#' @family estimator keys
prediction_keys <- function() {
  canned_estimator_lib$prediction_keys$PredictionKeys()
}

#' Enum for metric keys. 
#' 
#' Used by retrieving available metrics from canned estimators.
#' 
#' @examples 
#' metrics <- metric_keys()
#' 
#' # Get the available keys
#' names(metrics)
#' 
#' metrics$ACCURACY
#' 
#' @export
metric_keys <- function() {
  canned_estimator_lib$metric_keys$MetricKeys()
}

