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
svm_classifier <- function(feature_columns,
                           example_id_column,
                           weight_column_name,
                           run_options = NULL,
                           ...)
{
  run_options <- run_options %||% run_options()

  # construct estimator accepting those columns
  svm_clf <- learn$SVM(
    feature_columns = feature_columns,
    example_id_column = example_id_column,
    weight_column_name = weight_column_name,
    model_dir = recipe$model_dir %||% run_options$model_dir,
    config = run_options$run_config
  )

  tf_model(
    c("svm", "classifier"),
    estimator = svm_clf
  )

}
