#' TensorFlow -- Support Vector Machines
#' 
#' Perform Support Vector Machines for binary classification using TensorFlow.
#' @export
svm_classifier <- function(feature_columns,
                           example_id_column,
                           weight_column_name = NULL,
                           model_dir = NULL,
                           config = NULL)
{
  estimator <- py_suppress_warnings(
    contrib_learn$SVM(
      feature_columns = feature_columns,
      example_id_column = example_id_column,
      weight_column_name = weight_column_name,
      model_dir = resolve_model_dir(model_dir),
      config = config
    )
  )

  tf_classifier(estimator, "svm")
}
