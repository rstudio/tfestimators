#' Generates parsing spec for `tf$parse_example` to be used with regressors
#' 
#' If users keep data in `tf$Example` format, they need to call `tf$parse_example` 
#' with a proper feature spec. There are two main things that this utility 
#' helps:
#'    * Users need to combine parsing spec of features with labels and weights (if
#' any) since they are all parsed from same `tf$Example` instance. This utility
#' combines these specs.
#'    * It is difficult to map expected label by a regressor such as `dnn_regressor`
#' to corresponding `tf$parse_example` spec. This utility encodes it by getting
#' related information from users (key, dtype).
#' 
#' @param feature_columns An iterable containing all feature columns. All items 
#'   should be instances of classes derived from `_FeatureColumn`.
#' @param label_key A string identifying the label. It means `tf$Example` stores 
#'   labels with this key.
#' @param label_dtype A `tf$dtype` identifies the type of labels. By default it 
#'   is `tf$float32`.
#' @param label_default used as label if label_key does not exist in given 
#'   `tf$Example`. By default default_value is none, which means 
#'   `tf$parse_example` will error out if there is any missing label.
#' @param label_dimension Number of regression targets per example. This is the 
#'   size of the last dimension of the labels and logits `Tensor` objects 
#'   (typically, these have shape `[batch_size, label_dimension]`).
#' @param weight_column A string or a `_NumericColumn` created by 
#'   `column_numeric` defining feature column representing 
#'   weights. It is used to down weight or boost examples during training. It 
#'   will be multiplied by the loss of the example. If it is a string, it is 
#'   used as a key to fetch weight tensor from the `features`. If it is a 
#'   `_NumericColumn`, raw tensor is fetched by key `weight_column$key`, then 
#'   `weight_column$normalizer_fn` is applied on it to get weight tensor.
#'   
#' @return A dict mapping each feature key to a `FixedLenFeature` or 
#'   `VarLenFeature` value.
#'   
#' @section Raises: 
#'   * ValueError: If label is used in `feature_columns`. 
#'   * ValueError: If weight_column is used in `feature_columns`. 
#'   * ValueError: If any of the given `feature_columns` is not a `_FeatureColumn` instance. 
#'   * ValueError: If `weight_column` is not a `_NumericColumn` instance. 
#'   * ValueError: if label_key is `NULL`.
#'   
#' @export
#' @family parsing utilities
regressor_parse_example_spec <- function(feature_columns, label_key, label_dtype = tf$float32, label_default = NULL, label_dimension = 1L, weight_column = NULL) {
  tf$estimator$regressor_parse_example_spec(
    feature_columns = ensure_nullable_list(feature_columns),
    label_key = label_key,
    label_dtype = label_dtype,
    label_default = label_default,
    label_dimension = label_dimension,
    weight_column = weight_column
  )
}

#' Generates parsing spec for `tf$parse_example` to be used with classifiers
#' 
#' If users keep data in TensorFlow Example format, they need to call `tf$parse_example` 
#' with a proper feature spec. There are two main things that this utility 
#' helps: 
#'   * Users need to combine parsing spec of features with labels and 
#' weights (if any) since they are all parsed from same `tf$Example` instance. 
#' This utility combines these specs. 
#'   * It is difficult to map expected label by
#' a classifier such as `dnn_classifier` to corresponding `tf$parse_example` spec. 
#' This utility encodes it by getting related information from users (key, 
#' dtype). 
#' 
#' @param feature_columns An iterable containing all feature columns. All items 
#'   should be instances of classes derived from `_FeatureColumn`.
#' @param label_key A string identifying the label. It means `tf$Example` stores 
#'   labels with this key.
#' @param label_dtype A `tf$dtype` identifies the type of labels. By default it 
#'   is `tf$int64`. If user defines a `label_vocabulary`, this should be set as 
#'   `tf$string`. `tf$float32` labels are only supported for binary 
#'   classification.
#' @param label_default used as label if label_key does not exist in given 
#'   `tf$Example`. An example usage: let's say `label_key` is 'clicked' and 
#'   `tf$Example` contains clicked data only for positive examples in following 
#'   format `key:clicked, value:1`. This means that if there is no data with key
#'   'clicked' it should count as negative example by setting 
#'   `label_deafault=0`. Type of this value should be compatible with 
#'   `label_dtype`.
#' @param weight_column A string or a numeric column created by 
#'   [column_numeric()] defining feature column representing 
#'   weights. It is used to down weight or boost examples during training. It 
#'   will be multiplied by the loss of the example. If it is a string, it is 
#'   used as a key to fetch weight tensor from the `features`. If it is a 
#'   numeric column, raw tensor is fetched by key `weight_column$key`, then 
#'   `weight_column$normalizer_fn` is applied on it to get weight tensor.
#'   
#' @return A dict mapping each feature key to a `FixedLenFeature` or 
#'   `VarLenFeature` value.
#'   
#' @section Raises: 
#'   * ValueError: If label is used in `feature_columns`. 
#'   * ValueError: If weight_column is used in `feature_columns`. 
#'   * ValueError: If any of the given `feature_columns` is not a feature column instance. 
#'   * ValueError: If `weight_column` is not a numeric column instance. 
#'   * ValueError: if label_key is `NULL`.
#'   
#' @export
#' @family parsing utilities
#' 
classifier_parse_example_spec <- function(feature_columns, label_key, label_dtype = tf$int64, label_default = NULL, weight_column = NULL) {
  tf$estimator$classifier_parse_example_spec(
    feature_columns = ensure_nullable_list(feature_columns),
    label_key = label_key,
    label_dtype = label_dtype,
    label_default = label_default,
    weight_column = weight_column
  )
}
