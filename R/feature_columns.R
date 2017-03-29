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
#' @export
column_with_keys <- function(column_name, keys, default_value = -1L, combiner = "sum", dtype = tf$string) {
  tf$contrib$layers$sparse_column_with_keys(
    column_name = column_name,
    keys = keys,
    default_value = default_value,
    combiner = combiner,
    dtype = dtype
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
#' @export
column_with_hash_bucket <- function(column_name, hash_bucket_size, combiner = "sum", dtype = tf$string) {
  tf$contrib$layers$sparse_column_with_hash_bucket(
    column_name = column_name,
    hash_bucket_size = hash_bucket_size,
    combiner = combiner,
    dtype = dtype
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
#' @export
column_real_valued <- function(column_name, dimension = 1L, default_value = NULL, dtype = tf$float32, normalizer = NULL) {
  tf$contrib$layers$real_valued_column(
    column_name = column_name,
    dimension = dimension,
    default_value = default_value,
    dtype = dtype,
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
#' @param trainable (Optional). Should the embedding be trainable. Default is True
#' 
#' @export
column_embedding <- function(sparse_id_column, dimension, combiner = "mean", initializer = NULL, ckpt_to_load_from = NULL, tensor_name_in_ckpt = NULL, max_norm = NULL, trainable = TRUE) {
  tf$contrib$layers$embedding_column(
    sparse_id_column = sparse_id_column,
    dimension = dimension,
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
#' @export
column_crossed <- function(columns, hash_bucket_size, combiner = "sum", ckpt_to_load_from = NULL, tensor_name_in_ckpt = NULL, hash_key = NULL) {
  tf$contrib$layers$crossed_column(
    columns = columns,
    hash_bucket_size = hash_bucket_size,
    combiner = combiner,
    ckpt_to_load_from = ckpt_to_load_from,
    tensor_name_in_ckpt = tensor_name_in_ckpt,
    hash_key = hash_key
  )
}

