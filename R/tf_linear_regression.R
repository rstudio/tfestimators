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
#' recipe <- tf_simple_recipe(mpg ~ drat, data = mtcars)
#' tf_linear_regression(recipe = recipe)
tf_linear_regression <- function(recipe,
                                 tf.options = tf_options(),
                                 ...)
{
  # extract feature columns
  feature_columns <- recipe$feature.columns
  if (is.function(feature_columns))
    feature_columns <- feature_columns()

  # construct estimator accepting those columns
  lr <- learn$LinearRegressor(
    feature_columns = feature_columns,
    model_dir       = recipe$model.dir %||% tf.options$model.dir,
    optimizer       = recipe$optimizer %||% tf.options$optimizer
  )

  # fit the model
  lr$fit(
    input_fn = recipe$input.fn,
    steps = tf.options$steps
  )

  tf_model(
    "linear_regression",
    estimator = lr,
    recipe = recipe
  )

}

# TODO
tf_linear_classification <- function(recipe,
                                     tf.options = tf_options(),
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
    model_dir = tf.options$model.dir,
    optimizer = tf.options$optimizer
  )

  # fit the model
  lc$fit(
    input_fn = recipe$input.fn,
    steps = tf.options$steps
  )

  tf_model(
    "linear_classification",
    estimator = lc,
    recipe = recipe
  )

}

