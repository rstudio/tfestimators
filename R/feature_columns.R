#' @export
feature_columns <- function(x, ...) {
  UseMethod("feature_columns")
}

#' Construct column placeholders from vectors in an R object
#' @export
feature_columns.default <- function(x, columns) {
  ensure_valid_column_names(x, columns)
  function() {
    lapply(columns, function(column_name) {
      column_values <- x[[column_name]]
      if (is.numeric(column_values)) {
        column_real_valued(column_name)
      } else if (is.factor(column_values)) {
        column_with_keys(column_name, keys = levels(column_values))
      } else if (is.character(column_values)) {
        column_with_hash_bucket(column_name)
      }
    })
  }
}


#' Creates a _SparseColumn with keys.
#' 
#' @param column_name A string defining sparse column name.
#' @param keys A list defining vocabulary. Must be castable to `dtype`.
#' @param default_value The value to use for out-of-vocabulary feature values. Default is -1.
#' @param combiner A string specifying how to reduce if the sparse column is multivalent.
#' Currently "mean", "sqrtn" and "sum" are supported, with "sum" the default. 
#' "sqrtn" often achieves good accuracy, in particular with bag-of-words columns.
#' * "sum": do not normalize features in the column
#' * "mean": do l1 normalization on features in the column
#' * "sqrtn": do l2 normalization on features in the column
#' For more information: `tf.embedding_lookup_sparse`.
#' @param dtype Type of features. Only integer and string are supported.
#' 
#' @family feature_column wrappers
#' 
#' @export
column_sparse_with_keys <- function(column_name, keys, default_value = -1L, combiner = "sum", dtype = tf$string) {
  contrib_feature_column_lib$sparse_column_with_keys(
    column_name = column_name,
    keys = keys,
    default_value = as_nullable_integer(default_value),
    combiner = combiner,
    dtype = check_dtype(dtype)
  )
}

#' Creates a _SparseColumn with hashed bucket configuration.
#' 
#' @param column_name A string defining sparse column name.
#' @param hash_bucket_size An int that is > 1. The number of buckets.
#' @param combiner A string specifying how to reduce if the sparse column is multivalent. 
#' Currently "mean", "sqrtn" and "sum" are supported, with "sum" the default. 
#' "sqrtn" often achieves good accuracy, in particular with bag-of-words columns.
#' * "sum": do not normalize features in the column
#' * "mean": do l1 normalization on features in the column
#' * "sqrtn": do l2 normalization on features in the column
#' 
#' For more information: `tf.embedding_lookup_sparse`.
#' 
#' @param dtype The type of features. Only string and integer types are supported.
#' 
#' @family feature_column wrappers
#' 
#' @export
column_sparse_with_hash_bucket <- function(column_name, hash_bucket_size, combiner = "sum", dtype = tf$string) {
  hash_bucket_size <- as.integer(hash_bucket_size)
  if (hash_bucket_size <= 1) {
    stop("hash_bucket_size must be larger than 1")
  }
  contrib_feature_column_lib$sparse_column_with_hash_bucket(
    column_name = column_name,
    hash_bucket_size = hash_bucket_size,
    combiner = combiner,
    dtype = check_dtype(dtype)
  )
}

#' Creates a `_RealValuedColumn` for dense numeric data.
#' 
#' @param column_name A string defining real valued column name.
#' @param dimension An integer specifying dimension of the real valued column.
#' The default is 1. When dimension is not None, the Tensor representing the _RealValuedColumn will have the shape of [batch_size, dimension].
#' A None dimension means the feature column should be treat as variable length and will be parsed as a `SparseTensor`.
#' @param default_value A single value compatible with dtype or a list of values compatible with dtype which the column takes on during tf.Example parsing 
#' if data is missing. When dimension is not None, a default value of None will cause tf.parse_example to fail if an example does not contain this column. 
#' If a single value is provided, the same value will be applied as the default value for every dimension. If a list of values is provided, 
#' the length of the list should be equal to the value of `dimension`. Only scalar default value is supported in case dimension is not specified.
#' @param dtype defines the type of values. Default value is tf.float32. Must be a non-quantized, real integer or floating point type.
#' @param normalizer If not None, a function that can be used to normalize the value of the real valued column after default_value is applied for parsing. 
#' Normalizer function takes the input tensor as its argument, and returns the output tensor. (e.g. lambda x: (x - 3.0) / 4.2). 
#' Note that for variable length columns, the normalizer should expect an input_tensor of type `SparseTensor`.
#' 
#' @family feature_column wrappers
#' 
#' @export
column_real_valued <- function(column_name, dimension = 1L, default_value = NULL, dtype = tf$float32, normalizer = NULL) {
  contrib_feature_column_lib$real_valued_column(
    column_name = column_name,
    dimension = as_nullable_integer(dimension),
    default_value = default_value,
    dtype = check_dtype(dtype),
    normalizer = normalizer
  )
}

#' Creates an `_EmbeddingColumn` for feeding sparse data into a DNN.
#' 
#' @param sparse_id_column A `_SparseColumn` which is created by for example `sparse_column_with_*` or crossed_column functions. Note that `combiner` defined in `sparse_id_column` is ignored.
#' @param dimension An integer specifying dimension of the embedding.
#' @param combiner A string specifying how to reduce if there are multiple entries in a single row. Currently "mean", "sqrtn" and "sum" are supported, with "mean" the default. 
#' "sqrtn" often achieves good accuracy, in particular with bag-of-words columns. Each of this can be thought as example level normalizations on the column: 
#' * "sum": do not normalize       
#' * "mean": do l1 normalization       
#' * "sqrtn": do l2 normalization     
#' For more information: `tf.embedding_lookup_sparse`.
#' @param initializer A variable initializer function to be used in embedding variable initialization. 
#' If not specified, defaults to `tf.truncated_normal_initializer` with mean 0.0 and standard deviation 1/sqrt(sparse_id_column.length).
#' @param ckpt_to_load_from (Optional). String representing checkpoint name/pattern to restore the column weights. 
#' Required if `tensor_name_in_ckpt` is not None.
#' @param tensor_name_in_ckpt (Optional). Name of the `Tensor` in the provided checkpoint from which to restore the column weights. 
#' Required if `ckpt_to_load_from` is not None.
#' @param max_norm (Optional). If not None, embedding values are l2-normalized to the value of max_norm.
#' @param trainable (Optional). Should the embedding be trainable. Default is True.
#' 
#' @family feature_column wrappers
#' 
#' @export
column_embedding <- function(sparse_id_column, dimension, combiner = "mean", initializer = NULL, ckpt_to_load_from = NULL, tensor_name_in_ckpt = NULL, max_norm = NULL, trainable = TRUE) {
  contrib_feature_column_lib$embedding_column(
    sparse_id_column = sparse_id_column,
    dimension = as.integer(dimension),
    combiner = combiner,
    initializer = initializer,
    ckpt_to_load_from = ckpt_to_load_from,
    tensor_name_in_ckpt = tensor_name_in_ckpt,
    max_norm = max_norm,
    trainable = trainable
  )
}

#' Creates a _CrossedColumn for performing feature crosses.
#' 
#' @param columns An iterable of _FeatureColumn. Items can be an instance of _SparseColumn, _CrossedColumn, or _BucketizedColumn.
#' @param hash_bucket_size An int that is > 1. The number of buckets.
#' @param combiner A string specifying how to reduce if there are multiple entries in a single row. Currently "mean", "sqrtn" and "sum" are supported, with "sum" the default. 
#' "sqrtn" often achieves good accuracy, in particular with bag-of-words columns. Each of this can be thought as example level normalizations on the column::       
#' * "sum": do not normalize       
#' * "mean": do l1 normalization       
#' * "sqrtn": do l2 normalization     
#' For more information: `tf.embedding_lookup_sparse`.
#' @param ckpt_to_load_from (Optional). String representing checkpoint name/pattern to restore the column weights. Required if `tensor_name_in_ckpt` is not None.
#' @param tensor_name_in_ckpt (Optional). Name of the `Tensor` in the provided checkpoint from which to restore the column weights. Required if `ckpt_to_load_from` is not None.
#' @param hash_key Specify the hash_key that will be used by the `FingerprintCat64` function to combine the crosses fingerprints on SparseFeatureCrossOp (optional).
#' 
#' @family feature_column wrappers
#' 
#' @export
column_crossed <- function(columns, hash_bucket_size, combiner = "sum", ckpt_to_load_from = NULL, tensor_name_in_ckpt = NULL, hash_key = NULL) {
  hash_bucket_size <- as.integer(hash_bucket_size)
  if (hash_bucket_size <= 1) {
    stop("hash_bucket_size must be larger than 1")
  }
  contrib_feature_column_lib$crossed_column(
    columns = columns,
    hash_bucket_size = hash_bucket_size,
    combiner = combiner,
    ckpt_to_load_from = ckpt_to_load_from,
    tensor_name_in_ckpt = tensor_name_in_ckpt,
    hash_key = hash_key
  )
}


#' Creates a _SparseColumn by combining sparse_id_column with a weight column.
#' 
#' Example: 
#' 
#' ```python 
#' sparse_feature = sparse_column_with_hash_bucket(column_name="sparse_col", hash_bucket_size=1000)
#' weighted_feature = weighted_sparse_column(sparse_id_column=sparse_feature, weight_column_name="weights_col")
#' ``` 
#' This configuration assumes that input dictionary of model contains the following two items: 
#' * (key="sparse_col", value=sparse_tensor) where sparse_tensor is a SparseTensor. 
#' * (key="weights_col", value=weights_tensor) where weights_tensor is a SparseTensor. 
#' Following are assumed to be true: 
#' * sparse_tensor.indices = weights_tensor.indices 
#' * sparse_tensor.dense_shape = weights_tensor.dense_shape
#' 
#' @param sparse_id_column A `_SparseColumn` which is created by `column_sparse_with_*` functions.
#' @param weight_column_name A string defining a sparse column name which represents weight or value of the corresponding sparse id feature.
#' @param dtype Type of weights, such as `tf.float32`. Only floating and integer weights are supported. Returns: A _WeightedSparseColumn composed of two sparse features: one represents id, the other represents weight (value) of the id feature in that example.
#' 
#' @return A _WeightedSparseColumn composed of two sparse features: one represents id, the other represents weight (value) of the id feature in that example. Raises: ValueError: if dtype is not convertible to float.
#' 
#' @section Raises:
#' ValueError: if dtype is not convertible to float.
#' 
#' @family feature_column wrappers
#' 
#' @export
column_sparse_weighted <- function(sparse_id_column, weight_column_name, dtype = tf$float32) {
  contrib_feature_column_lib$weighted_sparse_column(
    sparse_id_column = sparse_id_column,
    weight_column_name = weight_column_name,
    dtype = check_dtype(dtype)
  )
}

#' Creates an `_OneHotColumn` for a one-hot or multi-hot repr in a DNN.
#' 
#' @param sparse_id_column A _SparseColumn which is created by `column_sparse_with_*` or crossed_column functions. Note that `combiner` defined in `sparse_id_column` is ignored.
#' 
#' @return An _OneHotColumn.
#' 
#' @family feature_column wrappers
#' 
#' @export
column_one_hot <- function(sparse_id_column) {
  tf$contrib$layers$one_hot_column(
    sparse_id_column = sparse_id_column
  )
}


#' Creates a _BucketizedColumn for discretizing dense input.
#' 
#' 
#' @param source_column A _RealValuedColumn defining dense column.
#' @param boundaries A list or list of floats specifying the boundaries. It has to be sorted.
#' 
#' @return A _BucketizedColumn.
#' 
#' @family feature_column wrappers
#' 
#' @section Raises:
#' ValueError: if 'boundaries' is empty or not sorted.
#' 
#' @export
column_bucketized <- function(source_column, boundaries) {
  contrib_feature_column_lib$bucketized_column(
    source_column = source_column,
    boundaries = boundaries
  )
}

