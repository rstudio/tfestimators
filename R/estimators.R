#' @param model_dir Directory to save the model parameters, graph, and so on.
#'   This can also be used to load checkpoints from the directory into a
#'   estimator to continue training a previously saved model.
#'
#' @param label_dimension Number of regression targets per example. This is the 
#'   size of the last dimension of the labels and logits `Tensor` objects 
#'   (typically, these have shape `[batch_size, label_dimension]`).
#'
#' @param weight_column A string or a `_NumericColumn` created by
#'   [column_numeric()] defining feature column representing weights. It is used
#'   to down weight or boost examples during training. It will be multiplied by
#'   the loss of the example. If it is a string, it is used as a key to fetch
#'   weight tensor from the `features` argument. If it is a `_NumericColumn`,
#'   then the raw tensor is fetched by key `weight_column.key`, then
#'   `weight_column.normalizer_fn` is applied on it to get weight tensor.
#'   
#' @param config A `RunConfig` object, used to configure the runtime
#'   settings.
#'   
#' @param input_layer_partitioner An optional partitioner for the input layer.
#'   Defaults to `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
#' 
#' @name estimators
NULL
