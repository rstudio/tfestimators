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
                                 run.options = run_options(),
                                 skip_fit = FALSE,
                                 ...)
{
  # extract feature columns
  linear.feature.columns <- recipe$linear.feature.columns
  dnn.feature.columns <- recipe$dnn.feature.columns
  if (is.function(linear.feature.columns))
    linear.feature.columns <- linear.feature.columns()
  if (is.function(dnn.feature.columns))
    dnn.feature.columns <- dnn.feature.columns()

  lm_dnn_r <- learn$DNNLinearCombinedRegressor(
    linear_feature_columns = linear.feature.columns,
    dnn_feature_columns = dnn.feature.columns,
    model_dir = run.options$model.dir %||% run.options$model.dir,
    config = run.options$run.config,
    ...
  )

  if(!skip_fit) {
    # fit the model
    lm_dnn_r$fit(
      input_fn = recipe$input.fn,
      steps = run.options$steps
    )
  }

  tf_model(
    "linear_dnn_combined_regression",
    estimator = lm_dnn_r,
    recipe = recipe
  )
}

#' TensorFlow -- Linear DNN Combined Classification
#'
#' Perform Linear DNN Combined Classification using TensorFlow.
#'
#' @template roxlate-recipe
#' @template roxlate-tf-options
#' @template roxlate-tf-dots
#'
#' @export
#' @examples
#' recipe <- simple_linear_dnn_combined_recipe(mtcars, response = "mpg", linear.features = c("cyl"), dnn.features = c("drat"))
#' linear_dnn_combined_classification(recipe = recipe, dnn_hidden_units = c(10L, 10L, 10L))
linear_dnn_combined_classification <- function(recipe,
                                           run.options = run_options(),
                                           skip_fit = FALSE,
                                           ...)
{
  # extract feature columns
  linear.feature.columns <- recipe$linear.feature.columns
  dnn.feature.columns <- recipe$dnn.feature.columns
  if (is.function(linear.feature.columns))
    linear.feature.columns <- linear.feature.columns()
  if (is.function(dnn.feature.columns))
    dnn.feature.columns <- dnn.feature.columns()

  lm_dnn_c <- learn$DNNLinearCombinedClassifier(
    linear_feature_columns = linear.feature.columns,
    dnn_feature_columns = dnn.feature.columns,
    model_dir = run.options$model.dir %||% run.options$model.dir,
    config = run.options$run.config,
    ...
  )

  if(!skip_fit) {
    # fit the model
    lm_dnn_c$fit(
      input_fn = recipe$input.fn,
      steps = run.options$steps
    )
  }

  tf_model(
    "linear_dnn_combined_classification",
    estimator = lm_dnn_c,
    recipe = recipe
  )
}
