#' TensorFlow -- Linear Regression
#'
#' Perform linear regression using TensorFlow.
#'
#' @template roxlate-recipe
#' @template roxlate-run-options
#' @template roxlate-tf-dots
#'
#' @export
linear_regressor <- function(feature_columns,
                             run_options = NULL,
                              ...)
{
  run_options <- run_options %||% run_options()
  
  # extract feature columns
  feature_columns <- resolve_fn(feature_columns)

  # construct estimator accepting those columns
  lr <- learn$LinearRegressor(
    feature_columns = feature_columns,
    model_dir = run_options$model_dir,
    config = run_options$run_config,
    ...
  )

  tf_model(
    c("linear", "regressor"),
    estimator = lr
  )

}

#' TensorFlow -- Linear Classification
#'
#' Perform linear classification using TensorFlow.
#'
#' @template roxlate-recipe
#' @template roxlate-run-options
#' @template roxlate-tf-dots
#'
#' @export
linear_classifier <- function(feature_columns,
                              n_classes = 2L,
                              run_options = NULL,
                              ...)
{
  run_options <- run_options %||% run_options()
  
  # extract feature columns
  feature_columns <- resolve_fn(feature_columns)

  # construct estimator accepting those columns
  lc <- learn$LinearClassifier(
    feature_columns = feature_columns,
    n_classes = n_classes,
    model_dir = run_options$model_dir,
    config = run_options$run_config,
    ...
  )

  tf_model(
    c("linear", "classifier"),
    estimator = lc
  )
}

