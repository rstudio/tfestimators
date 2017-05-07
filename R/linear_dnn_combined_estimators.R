#' Linear DNN Combined Regression
#'
#' Perform Linear DNN Combined Regression using TensorFlow.
#'
#' @export
#' @family canned estimators
linear_dnn_combined_regressor <- function(
  linear_feature_columns,
  dnn_feature_columns,
  model_dir = NULL,
  config = NULL,
  ...)
{

  # extract feature columns
  linear_feature_columns <- resolve_fn(linear_feature_columns)
  dnn_feature_columns <- resolve_fn(dnn_feature_columns)

  lm_dnn_r <- learn$DNNLinearCombinedRegressor(
    linear_feature_columns = linear_feature_columns,
    dnn_feature_columns = dnn_feature_columns,
    model_dir = model_dir,
    config = config,
    ...
  )

  tf_model(
    c("linear_dnn_combined", "regressor"),
    estimator = lm_dnn_r
  )
}

#' Linear DNN Combined Classification
#'
#' Perform Linear DNN Combined Classification using TensorFlow.
#'
#' @export
#' @family canned estimators
linear_dnn_combined_classifier <- function(
  linear_feature_columns,
  dnn_feature_columns,
  model_dir = NULL,
  config = NULL,
  ...)
{

  linear_feature_columns <- resolve_fn(linear_feature_columns)
  dnn_feature_columns <- resolve_fn(dnn_feature_columns)

  lm_dnn_c <- learn$DNNLinearCombinedClassifier(
    linear_feature_columns = linear_feature_columns,
    dnn_feature_columns = dnn_feature_columns,
    model_dir = model_dir,
    config = config,
    ...
  )

  tf_model(
    c("linear_dnn_combined", "classifier"),
    estimator = lm_dnn_c
  )
}
