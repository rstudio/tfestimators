#' An estimator for TensorFlow Linear regression problems.
#' 
#' Train a linear regression model to predict label value given observation of 
#' feature values.
#' 
#' @param feature_columns An iterable containing all the feature columns used by
#'   the model. All items in the set should be instances of classes derived from
#'   `FeatureColumn`.
#' @param model_dir Directory to save model parameters, graph and etc. This can
#'   also be used to load checkpoints from the directory into a estimator to
#'   continue training a previously saved model.
#' @param label_dimension Number of regression targets per example. This is the
#'   size of the last dimension of the labels and logits `Tensor` objects
#'   (typically, these have shape `[batch_size, label_dimension]`).
#' @param weight_column A string or a `_NumericColumn` created by
#'   `numeric_column` defining feature column representing weights. It is used
#'   to down weight or boost examples during training. It will be multiplied by
#'   the loss of the example. If it is a string, it is used as a key to fetch
#'   weight tensor from the `features`. If it is a `_NumericColumn`, raw tensor
#'   is fetched by key `weight_column.key`, then weight_column.normalizer_fn is
#'   applied on it to get weight tensor.
#' @param optimizer An instance of `tf.Optimizer` used to train the model.
#'   Defaults to FTRL optimizer.
#' @param config `RunConfig` object to configure the runtime settings.
#' @param partitioner Optional. Partitioner for input layer.
#'   
#' @export
#' @family canned estimators
linear_regressor <- function(feature_columns,
                             model_dir = NULL,
                             label_dimension = 1L,
                             weight_column = NULL,
                             optimizer = "Ftrl",
                             config = NULL,
                             partitioner = NULL)
{
  estimator <- py_suppress_warnings(
    tf$estimator$LinearRegressor(
      feature_columns = feature_columns,
      model_dir = model_dir,
      weight_column = weight_column,
      optimizer = optimizer,
      config = config,
      partitioner = partitioner,
      label_dimension = as.integer(label_dimension)
    )
  )

  tf_regressor(estimator, "linear_regressor")
}

#' Linear classifier model.
#' 
#' Train a linear model to classify instances into one of multiple possible 
#' classes. When number of possible classes is 2, this is binary classification.
#' 
#' @param feature_columns An iterable containing all the feature columns used by
#'   the model. All items in the set should be instances of classes derived from
#'   `FeatureColumn`.
#' @param model_dir Directory to save model parameters, graph and etc. This can 
#'   also be used to load checkpoints from the directory into a estimator to 
#'   continue training a previously saved model.
#' @param n_classes number of label classes. Default is binary classification. 
#'   Note that class labels are integers representing the class index (i.e. 
#'   values from 0 to n_classes-1). For arbitrary label values (e.g. string 
#'   labels), convert to class indices first.
#' @param weight_column A string or a `_NumericColumn` created by 
#'   `numeric_column` defining feature column representing weights. It is used
#'   to down weight or boost examples during training. It will be multiplied by
#'   the loss of the example. If it is a string, it is used as a key to fetch
#'   weight tensor from the `features`. If it is a `_NumericColumn`, raw tensor
#'   is fetched by key `weight_column.key`, then weight_column.normalizer_fn is
#'   applied on it to get weight tensor.
#' @param label_vocabulary A list of strings represents possible label values. 
#'   If given, labels must be string type and have any value in 
#'   `label_vocabulary`. If it is not given, that means labels are already 
#'   encoded as integer or float within [0, 1] for `n_classes=2` and encoded as 
#'   integer values in {0, 1,..., n_classes-1} for `n_classes`>2 . Also there 
#'   will be errors if vocabulary is not provided and labels are string.
#' @param optimizer An instance of `tf.Optimizer` used to train the model. 
#'   Defaults to FTRL optimizer.
#' @param config `RunConfig` object to configure the runtime settings.
#' @param partitioner Optional. Partitioner for input layer.
#'   
#' @export
#' @family canned estimators
linear_classifier <- function(feature_columns,
                              model_dir = NULL,
                              n_classes = 2L,
                              weight_column = NULL,
                              label_vocabulary = NULL,
                              optimizer = "Ftrl",
                              config = NULL,
                              partitioner = NULL)
{
  estimator <- py_suppress_warnings(
    tf$estimator$LinearClassifier(
      feature_columns = feature_columns,
      model_dir = model_dir,
      n_classes = as.integer(n_classes),
      weight_column = weight_column,
      label_vocabulary = label_vocabulary,
      optimizer = optimizer,
      config = config,
      partitioner = partitioner
    )
  )

  tf_classifier(estimator, "linear_classifier")
}
