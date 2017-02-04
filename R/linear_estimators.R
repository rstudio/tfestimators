#' TensorFlow -- Linear Regression
#'
#' Perform linear regression using TensorFlow.
#'
#' @template roxlate-recipe
#' @template roxlate-tf-options
#' @template roxlate-tf-dots
#'
#' @export
#' @examples
#' recipe <- simple_linear_recipe(mpg ~ drat, data = mtcars)
#' linear_regression(recipe = recipe)
linear_regression <- function(recipe,
                              run.options = run_options(),
                              skip_fit = FALSE,
                              ...)
{
  # extract feature columns
  feature_columns <- recipe$feature.columns
  if (is.function(feature_columns))
    feature_columns <- feature_columns()

  # construct estimator accepting those columns
  lr <- learn$LinearRegressor(
    feature_columns = feature_columns,
    model_dir = recipe$model.dir %||% run.options$model.dir,
    config = run.options$run.config,
    ...
  )

  if(!skip_fit) {
    # fit the model
    lr$fit(
      input_fn = recipe$input.fn,
      steps = run.options$steps
    )
  }

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
#' @template roxlate-tf-options
#' @template roxlate-tf-dots
#'
#' @export
#' @examples
#' recipe <- simple_linear_recipe(mpg ~ drat, data = mtcars)
#' linear_classification(recipe = recipe)
linear_classification <- function(recipe,
                                  run.options = run_options(),
                                  skip_fit = FALSE,
                                  ...)
{
  # extract feature columns
  feature_columns <- recipe$feature.columns
  if (is.function(feature_columns))
    feature_columns <- feature_columns()

  # construct estimator accepting those columns
  lc <- learn$LinearClassifier(
    feature_columns = feature_columns,
    n_classes = recipe$n.classes %||% 2L,
    model_dir = run.options$model.dir,
    config = run.options$run.config,
    ...
  )

  if(!skip_fit) {
    # fit the model
    lc$fit(
      input_fn = recipe$input.fn,
      steps = run.options$steps
    )
  }

  tf_model(
    "linear_classification",
    estimator = lc,
    recipe = recipe
  )
}

