#' TensorFlow -- Linear DNN Combined Regression
#'
#' Perform Linear DNN Combined Regression using TensorFlow.
#'
#' @export
linear_dnn_combined_regressor <- function(
  linear_feature_columns,
  dnn_feature_columns,
  run_options = NULL,
  ...)
{
  run_options <- run_options %||% run_options()
  
  # extract feature columns
  linear_feature_columns <- resolve_fn(linear_feature_columns)
  dnn_feature_columns <- resolve_fn(dnn_feature_columns)

  lm_dnn_r <- learn$DNNLinearCombinedRegressor(
    linear_feature_columns = linear_feature_columns,
    dnn_feature_columns = dnn_feature_columns,
    model_dir = run_options$model_dir %||% run_options$model_dir,
    config = run_options$run_config,
    ...
  )

  tf_model(
    c("linear_dnn_combined", "regressor"),
    estimator = lm_dnn_r
  )
}

#' TensorFlow -- Linear DNN Combined Classification
#'
#' Perform Linear DNN Combined Classification using TensorFlow.
#'
#' @export
linear_dnn_combined_classifier <- function(
  linear_feature_columns,
  dnn_feature_columns,
  run_options = NULL,
  ...)
{
  run_options <- run_options %||% run_options()
  
  linear_feature_columns <- resolve_fn(linear_feature_columns)
  dnn_feature_columns <- resolve_fn(dnn_feature_columns)

  lm_dnn_c <- learn$DNNLinearCombinedClassifier(
    linear_feature_columns = linear_feature_columns,
    dnn_feature_columns = dnn_feature_columns,
    model_dir = run_options$model_dir %||% run_options$model_dir,
    config = run_options$run_config,
    ...
  )

  tf_model(
    c("linear_dnn_combined", "classifier"),
    estimator = lm_dnn_c
  )
}
