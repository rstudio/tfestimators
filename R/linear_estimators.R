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
  args <- as.list(environment(), all.names = TRUE)
  
  estimator <- py_suppress_warnings(
    tf$estimator$LinearRegressor(
      feature_columns = ensure_nullable_list(feature_columns),
      model_dir = resolve_model_dir(model_dir),
      weight_column = cast_nullable_string(weight_column),
      optimizer = optimizer,
      config = config,
      partitioner = partitioner,
      label_dimension = cast_scalar_integer(label_dimension)
    )
  )

  new_tf_regressor(estimator, args = args, 
                   subclass = "tf_estimator_regressor_linear_regressor")
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
  args <- as.list(environment(), all.names = TRUE)

  estimator <- py_suppress_warnings(
    tf$estimator$LinearClassifier(
      feature_columns = ensure_nullable_list(feature_columns),
      model_dir = resolve_model_dir(model_dir),
      n_classes = cast_scalar_integer(n_classes),
      weight_column = cast_nullable_string(weight_column),
      label_vocabulary = label_vocabulary,
      optimizer = optimizer,
      config = config,
      partitioner = partitioner
    )
  )

  new_tf_classifier(estimator, args = args,
                    subclass = "tf_estimator_classifier_linear_classifier")
}
