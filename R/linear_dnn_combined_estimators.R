#' @family recipes
#' @export
linear_dnn_combined_recipe <- function(linear.feature.columns, dnn.feature.columns, input.fn, run.config, ...) {
  
  object <- list(
    linear.feature.columns = linear.feature.columns,
    dnn.feature.columns = dnn.feature.columns,
    input.fn = input.fn,
    run.config = run.config,
    ...
  )
  
  class(object) <- "linear_dnn_combined_recipe"
  object
}

# TODO:
#' @export
#' @family recipes
#' @examples
#' # two ways of constructing the same recipe
#' simple_linear_dnn_combined_recipe(mpg ~ cyl, data = mtcars)
#' simple_linear_dnn_combined_recipe(mtcars, response = "mpg", features = c("cyl"))
simple_linear_dnn_combined_recipe <- function(x, ...) {
  UseMethod("simple_linear_dnn_combined_recipe")
}

# TODO: Interface like simple_linear_dnn_combined_recipe(mpg ~ linear(cyl) + dnn(drat) )
#' #' @export
#' simple_linear_dnn_combined_recipe.formula <- function(x, data, ...) {
#'   parsed <- parse_formula(x)
#'   simple_linear_dnn_combined_recipe(data, parsed$response, parsed$features)
#' }

#' @export
simple_linear_dnn_combined_recipe.default <- function(x, response, linear.features, dnn.features, run.config, ...) {
  
  linear.feature.columns <- function() {
    tf_auto_inferred_columns(x, linear.features)
  }
  
  dnn.feature.columns <- function() {
    tf_auto_inferred_columns(x, dnn.features)
  }
  
  features <- c(linear.features, dnn.features)
  input.fn <- function() {
    feature_columns <- lapply(features, function(feature) {
      tf$constant(x[[feature]])
    })
    names(feature_columns) <- features
    response_column <- tf$constant(x[[response]])
    list(feature_columns, response_column)
  }
  
  run.config <- learn$RunConfig(tf_random_seed=1)
  
  linear_dnn_combined_recipe(linear.feature.columns, dnn.feature.columns, input.fn, run.config)
}

#' TensorFlow -- Linear DNN Combined Regression
#'
#' Perform Linear DNN Combined Regression using TensorFlow.
#'
#' @template roxlate-recipe
#' @template roxlate-tf-options
#' @template roxlate-tf-dots
#'
#' @export
#' @examples
#' recipe <- simple_linear_dnn_combined_recipe(mtcars, response = "mpg", linear.features = c("cyl"), dnn.features = c("drat"))
#' linear_dnn_combined_regression(recipe = recipe, dnn_hidden_units = c(10L, 10L, 10L))
linear_dnn_combined_regression <- function(recipe,
                                 tf.options = tf_options(),
                                 ...)
{
  # extract feature columns
  linear.feature.columns <- recipe$linear.feature.columns
  dnn.feature.columns <- recipe$dnn.feature.columns
  if (is.function(linear.feature.columns))
    linear.feature.columns <- linear.feature.columns()
  if (is.function(dnn.feature.columns))
    dnn.feature.columns <- dnn.feature.columns()
  # TODO: Decide whether this should be part of tf.options
  args <- list(...)
  if(! "dnn_hidden_units" %in% names(args)) stop("dnn_hidden_units must be provided")

  # construct estimator accepting those columns
  # TODO: This type of regressor has a lot of parameters you can specify, e.g. weight column, dropout, biases
  lm_dnn_r <- learn$DNNLinearCombinedRegressor(
    linear_feature_columns = linear.feature.columns,
    dnn_feature_columns = dnn.feature.columns,
    model_dir       = recipe$model.dir %||% tf.options$model.dir,
    ...
  )

  # fit the model
  lm_dnn_r$fit(
    input_fn = recipe$input.fn,
    steps = tf.options$steps
  )

  tf_model(
    "linear_dnn_combined_regression",
    estimator = lm_dnn_r,
    recipe = recipe
  )
}
