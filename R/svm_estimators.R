#' TensorFlow -- Support Vector Machines
#'
#' Perform Support Vector Machines for binary classification using TensorFlow.
#'
#' @template roxlate-recipe
#' @template roxlate-run-options
#' @template roxlate-tf-dots
#'
#' @export
#' @examples
#' recipe <- simple_linear_recipe(mpg ~ drat, data = mtcars)
#' svm_classifier(recipe = recipe)
svm_classification <- function(recipe,
                               run_options = NULL,
                               ...)
{
  run_options <- run_options %||% run_options()
  
  # extract feature columns
  feature_columns <- resolve_fn(recipe$feature_columns)
  
  # construct estimator accepting those columns
  lr <- learn$SVM(
    feature_columns = feature_columns,
    example_id_column = recipe$example_id_column,
    weight_column_name = recipe$weight_column_name,
    model_dir = recipe$model_dir %||% run_options$model_dir,
    config = run_options$run_config
  )
  
  tf_model(
    "linear_regression",
    estimator = lr,
    recipe = recipe
  )
  
}
