#' TensorFlow -- Linear DNN Combined Regression
#'
#' Perform Linear DNN Combined Regression using TensorFlow.
#'
#' @template roxlate-recipe
#' @template roxlate-run-options
#' @template roxlate-tf-dots
#'
#' @export
#' @examples
#' recipe <- simple_linear_dnn_combined_recipe(mtcars, response = "mpg", linear_features = c("cyl"), dnn_features = c("drat"))
#' linear_dnn_combined_regressor(recipe = recipe, dnn_hidden_units = c(10L, 10L, 10L))
linear_dnn_combined_regressor <- function(
  recipe,
  run_options = NULL,
  ...)
{
  run_options <- run_options %||% run_options()
  
  # extract feature columns
  linear_feature_columns <- resolve_fn(recipe$linear_feature_columns)
  dnn_feature_columns <- resolve_fn(recipe$dnn_feature_columns)

  lm_dnn_r <- learn$DNNLinearCombinedRegressor(
    linear_feature_columns = linear_feature_columns,
    dnn_feature_columns = dnn_feature_columns,
    model_dir = run_options$model_dir %||% run_options$model_dir,
    config = run_options$run_config,
    ...
  )

  tf_model(
    c("linear_dnn_combined", "regressor"),
    estimator = lm_dnn_r,
    recipe = recipe
  )
}

#' TensorFlow -- Linear DNN Combined Classification
#'
#' Perform Linear DNN Combined Classification using TensorFlow.
#'
#' @template roxlate-recipe
#' @template roxlate-run-options
#' @template roxlate-tf-dots
#'
#' @export
#' @examples
#' recipe <- simple_linear_dnn_combined_recipe(mtcars, response = "mpg", linear_features = c("cyl"), dnn_features = c("drat"))
#' linear_dnn_combined_classifier(recipe = recipe, dnn_hidden_units = c(10L, 10L, 10L))
linear_dnn_combined_classifier <- function(
  recipe,
  run_options = NULL,
  ...)
{
  run_options <- run_options %||% run_options()
  
  # extract feature columns
  linear_feature_columns <- recipe$linear_feature_columns
  dnn_feature_columns <- recipe$dnn_feature_columns
  if (is.function(linear_feature_columns))
    linear_feature_columns <- linear_feature_columns()
  if (is.function(dnn_feature_columns))
    dnn_feature_columns <- dnn_feature_columns()

  lm_dnn_c <- learn$DNNLinearCombinedClassifier(
    linear_feature_columns = linear_feature_columns,
    dnn_feature_columns = dnn_feature_columns,
    model_dir = run_options$model_dir %||% run_options$model_dir,
    config = run_options$run_config,
    ...
  )

  tf_model(
    c("linear_dnn_combined", "classifier"),
    estimator = lm_dnn_c,
    recipe = recipe
  )
}
