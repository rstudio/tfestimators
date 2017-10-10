#' Construct a Linear Estimator
#'
#' Construct a linear model, which can be used to predict a continuous outcome
#' (in the case of `linear_regressor()`) or a categorical outcome (in the case
#' of `linear_classifier()`).
#'
#' @inheritParams estimators
#' 
#' @param optimizer Either the name of the optimizer to be used when training
#'   the model, or a TensorFlow optimizer instance. Defaults to the FTRL
#'   optimizer.
#'
#' @family canned estimators
#' @name linear_estimators
NULL

#' @inheritParams linear_estimators
#' @name linear_estimators
#' @export
linear_regressor <- function(feature_columns,
                             model_dir = NULL,
                             label_dimension = 1L,
                             weight_column = NULL,
                             optimizer = "Ftrl",
                             config = NULL,
                             partitioner = NULL)
{
  args <- as.list(environment(), all = TRUE)
  
  estimator <- py_suppress_warnings(
    tf$estimator$LinearRegressor(
      feature_columns = ensure_nullable_list(feature_columns),
      model_dir = resolve_model_dir(model_dir),
      weight_column = weight_column,
      optimizer = optimizer,
      config = config,
      partitioner = partitioner,
      label_dimension = as.integer(label_dimension)
    )
  )

  tf_regressor(estimator, "linear_regressor", args)
}

#' @inheritParams linear_estimators
#' @name linear_estimators
#' @export
linear_classifier <- function(feature_columns,
                              model_dir = NULL,
                              n_classes = 2L,
                              weight_column = NULL,
                              label_vocabulary = NULL,
                              optimizer = "Ftrl",
                              config = NULL,
                              partitioner = NULL)
{
  args <- as.list(environment(), all = TRUE)

  estimator <- py_suppress_warnings(
    tf$estimator$LinearClassifier(
      feature_columns = ensure_nullable_list(feature_columns),
      model_dir = resolve_model_dir(model_dir),
      n_classes = as.integer(n_classes),
      weight_column = weight_column,
      label_vocabulary = label_vocabulary,
      optimizer = optimizer,
      config = config,
      partitioner = partitioner
    )
  )

  tf_classifier(estimator, "linear_classifier", args)
}
