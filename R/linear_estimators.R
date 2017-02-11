#' TensorFlow -- Linear Regression
#'
#' Perform linear regression using TensorFlow.
#'
#' @template roxlate-recipe
#' @template roxlate-run-options
#' @template roxlate-tf-dots
#'
#' @export
#' @examples
#' recipe <- simple_linear_recipe(mpg ~ drat, data = mtcars)
#' linear_regression(recipe = recipe)
linear_regression <- function(recipe,
                              run_options = NULL,
                              ...)
{
  run_options <- run_options %||% run_options()
  
  # extract feature columns
  feature_columns <- resolve_fn(recipe$feature_columns)

  # construct estimator accepting those columns
  lr <- learn$LinearRegressor(
    feature_columns = feature_columns,
    model_dir = recipe$model_dir %||% run_options$model_dir,
    config = run_options$run_config,
    ...
  )

  tf_model(
    "linear_regression",
    estimator = lr,
    recipe = recipe
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
#' @examples
#' recipe <- simple_linear_recipe(mpg ~ drat, data = mtcars)
#' linear_classification(recipe = recipe)
linear_classification <- function(recipe,
                                  run_options = NULL,
                                  ...)
{
  run_options <- run_options %||% run_options()
  
  # extract feature columns
  feature_columns <- resolve_fn(recipe$feature_columns)

  # construct estimator accepting those columns
  lc <- learn$LinearClassifier(
    feature_columns = feature_columns,
    n_classes = recipe$n.classes %||% 2L,
    model_dir = run_options$model_dir,
    config = run_options$run_config,
    ...
  )

  tf_model(
    "linear_classification",
    estimator = lc,
    recipe = recipe
  )
}

