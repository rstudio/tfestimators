#' TensorFlow -- Support Vector Machines
#'
#' Perform Support Vector Machines for binary classification using TensorFlow.
#' @export
svm_classifier <- function(feature_columns,
                           example_id_column,
                           weight_column_name = NULL,
                           run_options = NULL,
                           ...)
{
  run_options <- run_options %||% run_options()

  # extract feature columns
  feature_columns <- resolve_fn(feature_columns)

  svm_clf <- learn$SVM(
    feature_columns = feature_columns,
    example_id_column = example_id_column,
    weight_column_name = weight_column_name,
    model_dir = run_options$model_dir,
    config = run_options$run_config,
    ...
  )

  tf_model(
    c("svm", "classifier"),
    estimator = svm_clf
  )
}
