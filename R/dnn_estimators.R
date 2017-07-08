#' A regressor for TensorFlow DNN models.
#' 
#' @param hidden_units Iterable of number hidden units per layer. All layers are
#'   fully connected. Ex. `[64, 32]` means first layer has 64 nodes and second 
#'   one has 32.
#' @param feature_columns An iterable containing all the feature columns used by
#'   the model. All items in the set should be instances of classes derived from
#'   `_FeatureColumn`.
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
#'   Defaults to Adagrad optimizer.
#' @param activation_fn Activation function applied to each layer. If `NULL`, 
#'   will use `tf$nn$relu`.
#' @param dropout When not `NULL`, the probability we will drop out a given 
#'   coordinate.
#' @param input_layer_partitioner Optional. Partitioner for input layer. 
#'   Defaults to `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
#' @param config `RunConfig` object to configure the runtime settings.
#'   
#' @export
#' @family canned estimators
dnn_regressor <- function(hidden_units, feature_columns, model_dir = NULL, label_dimension = 1L, weight_column = NULL, optimizer = "Adagrad", activation_fn = relu, dropout = NULL, input_layer_partitioner = NULL, config = NULL)
{
  # construct estimator accepting those columns
  dnn_model <- tf$estimator$DNNRegressor(
    hidden_units = as.integer(hidden_units),
    feature_columns = feature_columns,
    model_dir = model_dir,
    label_dimension = as.integer(label_dimension),
    weight_column = weight_column,
    optimizer = optimizer,
    activation_fn = activation_fn,
    dropout = dropout,
    input_layer_partitioner = input_layer_partitioner,
    config = config
  )

  tf_model(
    c("dnn", "regressor"),
    estimator = dnn_model
  )

}

#' A classifier for TensorFlow DNN models.
#' 
#' @param hidden_units Iterable of number hidden units per layer. All layers are
#'   fully connected. Ex. `[64, 32]` means first layer has 64 nodes and second 
#'   one has 32.
#' @param feature_columns An iterable containing all the feature columns used by
#'   the model. All items in the set should be instances of classes derived from
#'   `_FeatureColumn`.
#' @param model_dir Directory to save model parameters, graph and etc. This can 
#'   also be used to load checkpoints from the directory into a estimator to 
#'   continue training a previously saved model.
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
#' @param optimizer An instance of `tf.Optimizer` used to train the model. 
#'   Defaults to Adagrad optimizer.
#' @param activation_fn Activation function applied to each layer. If `NULL`, 
#'   will use `tf$nn$relu`.
#' @param dropout When not `NULL`, the probability we will drop out a given 
#'   coordinate.
#' @param input_layer_partitioner Optional. Partitioner for input layer. 
#'   Defaults to `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
#' @param config `RunConfig` object to configure the runtime settings.
#'   
#' @export
#' @family canned estimators
dnn_classifier <- function(hidden_units, feature_columns, model_dir = NULL, n_classes = 2L, weight_column = NULL, label_vocabulary = NULL, optimizer = "Adagrad", activation_fn = relu, dropout = NULL, input_layer_partitioner = NULL, config = NULL)
{
  # construct estimator accepting those columns
  dnn_model <- tf$estimator$DNNClassifier(
    hidden_units = as.integer(hidden_units),
    feature_columns = feature_columns,
    model_dir = model_dir,
    n_classes = as.integer(n_classes),
    weight_column = weight_column,
    label_vocabulary = label_vocabulary,
    optimizer = optimizer,
    activation_fn = activation_fn,
    dropout = dropout,
    input_layer_partitioner = input_layer_partitioner,
    config = config
  )

  tf_model(
    c("dnn", "classifier"),
    estimator = dnn_model
  )
}

