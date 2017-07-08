#' An estimator for TensorFlow Linear and DNN joined models for regression.
#' 
#' Note: This estimator is also known as wide-n-deep.
#' 
#' @param model_dir Directory to save model parameters, graph and etc. This can 
#'   also be used to load checkpoints from the directory into a estimator to 
#'   continue training a previously saved model.
#' @param linear_feature_columns An iterable containing all the feature columns 
#'   used by linear part of the model. All items in the set must be instances of
#'   classes derived from `FeatureColumn`.
#' @param linear_optimizer An instance of `tf.Optimizer` used to apply gradients
#'   to the linear part of the model. Defaults to FTRL optimizer.
#' @param dnn_feature_columns An iterable containing all the feature columns 
#'   used by deep part of the model. All items in the set must be instances of 
#'   classes derived from `FeatureColumn`.
#' @param dnn_optimizer An instance of `tf.Optimizer` used to apply gradients to
#'   the deep part of the model. Defaults to Adagrad optimizer.
#' @param dnn_hidden_units List of hidden units per layer. All layers are fully 
#'   connected.
#' @param dnn_activation_fn Activation function applied to each layer. If NULL, 
#'   will use `tf$nn$relu`.
#' @param dnn_dropout When not NULL, the probability we will drop out a given 
#'   coordinate.
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
#' @param input_layer_partitioner Partitioner for input layer. Defaults to 
#'   `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
#' @param config RunConfig object to configure the runtime settings.
#'   
#' @export
#' @family canned estimators
dnn_linear_combined_regressor <- function(model_dir = NULL, linear_feature_columns = NULL, linear_optimizer = "Ftrl", dnn_feature_columns = NULL, dnn_optimizer = "Adagrad", dnn_hidden_units = NULL, dnn_activation_fn = tf$nn$relu, dnn_dropout = NULL, label_dimension = 1L, weight_column = NULL, input_layer_partitioner = NULL, config = NULL)
{
  dnn_linear_model <- tf$estimator$DNNLinearCombinedRegressor(
    model_dir = model_dir,
    linear_feature_columns = linear_feature_columns,
    linear_optimizer = linear_optimizer,
    dnn_feature_columns = dnn_feature_columns,
    dnn_optimizer = dnn_optimizer,
    dnn_hidden_units = as.integer(dnn_hidden_units),
    dnn_activation_fn = dnn_activation_fn,
    dnn_dropout = dnn_dropout,
    label_dimension = as.integer(label_dimension),
    weight_column = weight_column,
    input_layer_partitioner = input_layer_partitioner,
    config = config
  )

  tf_model(
    c("dnn_linear_combined", "regressor"),
    estimator = dnn_linear_model
  )
}

#' An estimator for TensorFlow Linear and DNN joined classification models.
#' 
#' Note: This estimator is also known as wide-n-deep.
#' 
#' @param model_dir Directory to save model parameters, graph and etc. This can 
#'   also be used to load checkpoints from the directory into a estimator to 
#'   continue training a previously saved model.
#' @param linear_feature_columns An iterable containing all the feature columns 
#'   used by linear part of the model. All items in the set must be instances of
#'   classes derived from `FeatureColumn`.
#' @param linear_optimizer An instance of `tf.Optimizer` used to apply gradients
#'   to the linear part of the model. Defaults to FTRL optimizer.
#' @param dnn_feature_columns An iterable containing all the feature columns 
#'   used by deep part of the model. All items in the set must be instances of 
#'   classes derived from `FeatureColumn`.
#' @param dnn_optimizer An instance of `tf.Optimizer` used to apply gradients to
#'   the deep part of the model. Defaults to Adagrad optimizer.
#' @param dnn_hidden_units List of hidden units per layer. All layers are fully 
#'   connected.
#' @param dnn_activation_fn Activation function applied to each layer. If NULL, 
#'   will use `tf$nn$relu`.
#' @param dnn_dropout When not NULL, the probability we will drop out a given 
#'   coordinate.
#' @param n_classes Number of label classes. Defaults to 2, namely binary 
#'   classification. Must be > 1.
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
#' @param input_layer_partitioner Partitioner for input layer. Defaults to 
#'   `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
#' @param config RunConfig object to configure the runtime settings.
#'   
#' @export
#' @family canned estimators
dnn_linear_combined_classifier <- function(model_dir = NULL, linear_feature_columns = NULL, linear_optimizer = "Ftrl", dnn_feature_columns = NULL, dnn_optimizer = "Adagrad", dnn_hidden_units = NULL, dnn_activation_fn = tf$nn$relu, dnn_dropout = NULL, n_classes = 2L, weight_column = NULL, label_vocabulary = NULL, input_layer_partitioner = NULL, config = NULL)
{
  dnn_linear_model <- tf$estimator$DNNLinearCombinedClassifier(
    model_dir = model_dir,
    linear_feature_columns = linear_feature_columns,
    linear_optimizer = linear_optimizer,
    dnn_feature_columns = dnn_feature_columns,
    dnn_optimizer = dnn_optimizer,
    dnn_hidden_units = as.integer(dnn_hidden_units),
    dnn_activation_fn = dnn_activation_fn,
    dnn_dropout = dnn_dropout,
    n_classes = as.integer(n_classes),
    weight_column = weight_column,
    label_vocabulary = label_vocabulary,
    input_layer_partitioner = input_layer_partitioner,
    config = config
  )

  tf_model(
    c("dnn_linear_combined", "classifier"),
    estimator = dnn_linear_model
  )
}
