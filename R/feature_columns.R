
#' Define feature columns
#'
#' @param names Character vector with names of all available features
#'  (or an R object that you can call `colnames()` or `names()` on 
#'  to extract names).
#'
#' @export
feature_columns <- function(names) {
  
  # capture names
  if (!is.character(names))
    names <- object_names(names)
  
  # return feature columns
  structure(
    class = "tfestimators_feature_columns", list(
    names = names,
    features = list()
  ))
}


#' A `_CategoricalColumn` with in-memory vocabulary.
#' 
#' Use this when your inputs are in string or integer format, and you have an
#' in-memory vocabulary mapping each value to an integer ID. By default,
#' out-of-vocabulary values are ignored. Use `default_value` to specify how to
#' include out-of-vocabulary values. For input dictionary `features`, `features$key` is either `Tensor` or
#' `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
#' and `''` for string. 
#' 
#' Note that these values are independent of the
#' `default_value` argument. 
#' 
#' @inheritParams column_numeric
#'
#' @param vocabulary_list An ordered iterable defining the vocabulary. 
#' Each feature is mapped to the index of its value (if present) in `vocabulary_list`. 
#' Must be castable to `dtype`.
#' @param dtype The type of features. Only string and integer types are supported. 
#' If `NULL`, it will be inferred from `vocabulary_list`.
#' @param default_value The value to use for values not in `vocabulary_list`.
#' 
#' @return A `_CategoricalColumn` with in-memory vocabulary.
#' 
#' @section Raises:
#' ValueError: if `vocabulary_list` is empty, or contains duplicate keys. ValueError: if `dtype` is not integer or string.
#' 
#' @export
#' @family feature_column wrappers
column_categorical_with_vocabulary_list <- function(fc, ..., vocabulary_list, dtype = NULL, default_value = -1L) {
  
  # select columns
  columns <- vars_select(fc$names, !!! quos(...))
  
  for (column in columns) {
    fc$features[[length(fc$features) + 1]] <- feature_column_lib$categorical_column_with_vocabulary_list(
      key = column,
      vocabulary_list = vocabulary_list,
      dtype = dtype,
      default_value = default_value
    )
  }
  
  fc
  
}

#' A `_CategoricalColumn` with a vocabulary file.
#' 
#' Use this when your inputs are in string or integer format, and you have a
#' vocabulary file that maps each value to an integer ID. By default,
#' out-of-vocabulary values are ignored. Use either (but not both) of
#' `num_oov_buckets` and `default_value` to specify how to include
#' out-of-vocabulary values. For input dictionary `features`, `features[key]` is either `Tensor` or
#' `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
#' and `''` for string. Note that these values are independent of the
#' `default_value` argument. Example with `num_oov_buckets`:
#' File '/us/states.txt' contains 50 lines, each with a 2-character U.S. state
#' abbreviation. All inputs with values in that file are assigned an ID 0-49,
#' corresponding to its line number. All other values are hashed and assigned an
#' ID 50-54.
#' 
#' @inheritParams column_numeric
#' 
#' @param vocabulary_file The vocabulary file name.
#' @param vocabulary_size Number of the elements in the vocabulary. This must be no greater than length of `vocabulary_file`, if less than length, later values are ignored.
#' @param num_oov_buckets Non-negative integer, the number of out-of-vocabulary buckets. All out-of-vocabulary inputs will be assigned IDs in the range `[vocabulary_size, vocabulary_size+num_oov_buckets)` based on a hash of the input value. A positive `num_oov_buckets` can not be specified with `default_value`.
#' @param default_value The integer ID value to return for out-of-vocabulary feature values, defaults to `-1`. This can not be specified with a positive `num_oov_buckets`.
#' @param dtype The type of features. Only string and integer types are supported.
#' 
#' @return A `_CategoricalColumn` with a vocabulary file.
#' 
#' @section Raises:
#' ValueError: `vocabulary_file` is missing.
#' ValueError: `vocabulary_size` is missing or < 1. 
#' ValueError: `num_oov_buckets` is not a non-negative integer. 
#' ValueError: `dtype` is neither string nor integer.
#' @family feature_column wrappers
#' 
#' @importFrom rlang enquo
#' @importFrom tidyselect vars_select quos
#' 
#' @export
column_categorical_with_vocabulary_file <- function(fc, ..., vocabulary_file, vocabulary_size, num_oov_buckets = 0L, default_value = NULL, dtype = tf$string) {
  
  # select columns
  columns <- vars_select(fc$names, !!! quos(...))
  
  for (column in columns) {
    fc$features[[length(fc$features) + 1]] <- feature_column_lib$categorical_column_with_vocabulary_file(
      key = column,
      vocabulary_file = vocabulary_file,
      vocabulary_size = vocabulary_size,
      num_oov_buckets = num_oov_buckets,
      default_value = default_value,
      dtype = dtype
    )
  }
  
  fc
}

#' A `_CategoricalColumn` that returns identity values.
#' 
#' Use this when your inputs are integers in the range `[0, num_buckets)`, and
#' you want to use the input value itself as the categorical ID. Values outside
#' this range will result in `default_value` if specified, otherwise it will
#' fail. 
#' 
#' Typically, this is used for contiguous ranges of integer indexes, but
#' it doesn't have to be. This might be inefficient, however, if many of IDs
#' are unused. Consider `categorical_column_with_hash_bucket` in that case. 
#' 
#' For input dictionary `features`, `features$key` is either `Tensor` or
#' `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
#' and `''` for string. Note that these values are independent of the
#' `default_value` argument.
#' 
#' @inheritParams column_numeric
#' 
#' @param num_buckets Range of inputs and outputs is `[0, num_buckets)`. If `NULL` then the range
#' is automatically determined by inspecting the data for unique values.
#' @param default_value If `NULL`, this column's graph operations will fail for out-of-range inputs. 
#' Otherwise, this value must be in the range `[0, num_buckets)`, and will replace inputs in that range.
#' 
#' @return A `_CategoricalColumn` that returns identity values.
#' 
#' @section Raises:
#' ValueError: if `num_buckets` is less than one. 
#' ValueError: if `default_value` is not in range `[0, num_buckets)`.
#' 
#' @family feature_column wrappers
#' @export
column_categorical_with_identity <- function(fc, ..., num_buckets = NULL, default_value = NULL) {
  
  # select columns
  columns <- vars_select(fc$names, !!! quos(...))
  
  for (column in columns) {
    
    if (!is.null(num_buckets))
      buckets <- num_buckets
    else
      buckets <- unique(fc$data[,column])
    
    fc$features[[length(fc$features) + 1]]  <- feature_column_lib$categorical_column_with_identity(
      key = column,
      num_buckets = buckets,
      default_value = default_value
    )
  }
  
  fc
}

#' Represents multi-hot representation of given categorical column.
#' 
#' Used to wrap any `categorical_column_*` (e.g., to feed to DNN). Use
#' `embedding_column` if the inputs are sparse. 
#' 
#' @param categorical_column A `_CategoricalColumn` which is created by `categorical_column_with_*` or `crossed_column` functions.
#' 
#' @return An `_IndicatorColumn`.
#' 
#' @export
column_indicator <- function(categorical_column) {
  feature_column_lib$indicator_column(
    categorical_column = categorical_column
  )
}

#' Represents sparse feature where ids are set by hashing.
#' 
#' Use this when your sparse features are in string or integer format, and you
#' want to distribute your inputs into a finite number of buckets by hashing.
#' output_id = Hash(input_feature_string) % bucket_size For input dictionary `features`, `features$key$` is either `Tensor` or
#' `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
#' and `''` for string. Note that these values are independent of the
#' `default_value` argument. 
#' 
#' @inheritParams column_numeric
#' 
#' @param hash_bucket_size An int > 1. The number of buckets.
#' @param dtype The type of features. Only string and integer types are supported.
#' 
#' @return A `_HashedCategoricalColumn`.
#' 
#' @section Raises:
#' ValueError: `hash_bucket_size` is not greater than 1. ValueError: `dtype` is neither string nor integer.
#' 
#' @family feature_column wrappers
#'   
#' @export
column_categorical_with_hash_bucket <- function(fc, ..., hash_bucket_size, dtype = tf$string) {
  hash_bucket_size <- as.integer(hash_bucket_size)
  if (hash_bucket_size <= 1) {
    stop("hash_bucket_size must be larger than 1")
  }
  
  # select columns
  columns <- vars_select(fc$names, !!! quos(...))
  
  
  for (column in columns) {
    fc$features[[length(fc$features) + 1]] <- feature_column_lib$categorical_column_with_hash_bucket(
      key = column,
      hash_bucket_size = hash_bucket_size,
      dtype = dtype
    )
  }
  
  fc
}

#' Represents real valued or numerical features.
#'
#' @param fc Feature columns
#' @param ... Expression(s) identifying input feature(s). Used as the column name
#'   and the dictionary key for feature parsing configs, feature `Tensor`
#'   objects, and feature columns.
#' @param shape An iterable of integers specifies the shape of the `Tensor`. An
#'   integer can be given which means a single dimension `Tensor` with given
#'   width. The `Tensor` representing the column will have the shape of
#'   [batch_size] + `shape`.
#' @param default_value A single value compatible with `dtype` or an iterable of
#'   values compatible with `dtype` which the column takes on during
#'   `tf.Example` parsing if data is missing. A default value of `NULL` will
#'   cause `tf.parse_example` to fail if an example does not contain this
#'   column. If a single value is provided, the same value will be applied as
#'   the default value for every item. If an iterable of values is provided, the
#'   shape of the `default_value` should be equal to the given `shape`.
#' @param dtype defines the type of values. Default value is `tf.float32`. Must
#'   be a non-quantized, real integer or floating point type.
#' @param normalizer_fn If not `NULL`, a function that can be used to normalize
#'   the value of the tensor after `default_value` is applied for parsing.
#'   Normalizer function takes the input `Tensor` as its argument, and returns
#'   the output `Tensor`. (e.g. lambda x: (x - 3.0) / 4.2). Please note that
#'   even though the most common use case of this function is normalization, it
#'   can be used for any kind of Tensorflow transformations.
#'
#' @return A `_NumericColumn`.
#'
#' @section Raises: TypeError: if any dimension in shape is not an int
#'   ValueError: if any dimension in shape is not a positive integer TypeError:
#'   if `default_value` is an iterable but not compatible with `shape`
#'   TypeError: if `default_value` is not compatible with `dtype`. ValueError:
#'   if `dtype` is not convertible to `tf.float32`.
#'
#' @family feature_column wrappers
#'
#' @export
column_numeric <- function(fc, ..., shape = list(1L), default_value = NULL, dtype = tf$float32, normalizer_fn = NULL) {
  
  # select columns
  columns <- vars_select(fc$names, !!! quos(...))
  
  # add them
  for (column in columns) {
    fc$features[[length(fc$features) + 1]] <- feature_column_lib$numeric_column(
      key = column,
      shape = shape,
      default_value = default_value,
      dtype = dtype,
      normalizer_fn = normalizer_fn
    )
  }
  
  # return feature columns for further piping
  fc
}

#' `_DenseColumn` that converts from sparse, categorical input.
#' 
#' Use this when your inputs are sparse, but you want to convert them to a dense
#' representation (e.g., to feed to a DNN). Inputs must be a `_CategoricalColumn` created by any of the
#' `categorical_column_*` function.
#' 
#' @param categorical_column A `_CategoricalColumn` created by a `categorical_column_with_*` function. 
#' This column produces the sparse IDs that are inputs to the embedding lookup.
#' @param dimension An integer specifying dimension of the embedding, must be > 0.
#' @param combiner A string specifying how to reduce if there are multiple entries in a single row. 
#' Currently 'mean', 'sqrtn' and 'sum' are supported, with 'mean' the default. 'sqrtn' often achieves good accuracy, in particular with bag-of-words columns.
#' Each of this can be thought as example level normalizations on the column. For more information, see `tf.embedding_lookup_sparse`.
#' @param initializer A variable initializer function to be used in embedding variable initialization. 
#' If not specified, defaults to `tf.truncated_normal_initializer` with mean `0.0` and standard deviation `1/sqrt(dimension)`.
#' @param ckpt_to_load_from String representing checkpoint name/pattern from which to restore column weights. 
#' Required if `tensor_name_in_ckpt` is not `NULL`.
#' @param tensor_name_in_ckpt Name of the `Tensor` in `ckpt_to_load_from` from which to restore the column weights. 
#' Required if `ckpt_to_load_from` is not `NULL`.
#' @param max_norm If not `NULL`, embedding values are l2-normalized to this value.
#' @param trainable Whether or not the embedding is trainable. Default is TRUE.
#' 
#' @return `_DenseColumn` that converts from sparse input.
#' 
#' @section Raises:
#' ValueError: if `dimension` not > 0. ValueError: if exactly one of `ckpt_to_load_from` and `tensor_name_in_ckpt` is specified. ValueError: if `initializer` is specified and is not callable.
#' 
#' @family feature_column wrappers
#'   
#' @export
column_embedding <- function(categorical_column, dimension, combiner = "mean", initializer = NULL, ckpt_to_load_from = NULL, tensor_name_in_ckpt = NULL, max_norm = NULL, trainable = TRUE) {
  feature_column_lib$embedding_column(
    categorical_column = categorical_column,
    dimension = as.integer(dimension),
    combiner = combiner,
    initializer = initializer,
    ckpt_to_load_from = ckpt_to_load_from,
    tensor_name_in_ckpt = tensor_name_in_ckpt,
    max_norm = max_norm,
    trainable = trainable
  )
}

#' Returns a column for performing crosses of categorical features.
#' 
#' Crossed features will be hashed according to `hash_bucket_size`. Conceptually,
#' the transformation can be thought of as: Hash(cartesian product of features) % `hash_bucket_size` For example, if the input features are:
#' * SparseTensor referred by first key: shape = [2, 2] [0, 0]: "a" [1, 0]: "b" [1, 1]: "c"
#' * SparseTensor referred by second key:
#' shape = [2, 1] [0, 0]: "d" [1, 0]: "e" then crossed feature will look like: shape = [2, 2] [0, 0]: Hash64("d", Hash64("a")) % hash_bucket_size [1, 0]: Hash64("e", Hash64("b")) % hash_bucket_size [1, 1]: Hash64("e", Hash64("c")) % hash_bucket_size
#' @param keys An iterable identifying the features to be crossed. Each element can be either: 
#' * string: Will use the corresponding feature which must be of string type. 
#' * `_CategoricalColumn`: Will use the transformed tensor produced by this column. Does not support hashed categorical column.
#' @param hash_bucket_size An int > 1. The number of buckets.
#' @param hash_key Specify the hash_key that will be used by the `FingerprintCat64` function to combine the crosses fingerprints on SparseCrossOp (optional).
#' 
#' @return A `_CrossedColumn`.
#' 
#' @section Raises:
#' ValueError: If `len(keys) < 2`. ValueError: If any of the keys is neither a string nor `_CategoricalColumn`. ValueError: If any of the keys is `_HashedCategoricalColumn`. ValueError: If `hash_bucket_size < 1`.
#' 
#' @family feature_column wrappers
#'   
#' @export
column_crossed <- function(keys, hash_bucket_size, hash_key = NULL) {
  hash_bucket_size <- as.integer(hash_bucket_size)
  if (hash_bucket_size <= 1) {
    stop("hash_bucket_size must be larger than 1")
  }
  feature_column_lib$crossed_column(
    keys = keys,
    hash_bucket_size = hash_bucket_size,
    hash_key = hash_key
  )
}


#' Applies weight values to a `_CategoricalColumn`.
#' 
#' Use this when each of your sparse inputs has both an ID and a value. For
#' example, if you're representing text documents as a collection of word
#' frequencies, you can provide 2 parallel sparse input features ('terms' and
#' 'frequencies' below). 
#' 
#' @param categorical_column A `_CategoricalColumn` created by `categorical_column_with_*` functions.
#' @param weight_feature_key String key for weight values.
#' @param dtype Type of weights, such as `tf.float32`. Only float and integer weights are supported.
#' 
#' @return A `_CategoricalColumn` composed of two sparse features: one represents id, the other represents weight (value) of the id feature in that example.
#' 
#' @section Raises:
#' ValueError: if `dtype` is not convertible to float.
#' 
#' @family feature_column wrappers
#'   
#' @export
column_weighted_categorical <- function(categorical_column, weight_feature_key, dtype = tf$float32) {
  feature_column_lib$weighted_categorical_column(
    categorical_column = categorical_column,
    weight_feature_key = weight_feature_key,
    dtype = dtype
  )
}


#' Represents discretized dense input.
#' 
#' Buckets include the left boundary, and exclude the right boundary. Namely,
#' `boundaries=[0., 1., 2.]` generates buckets `(-inf, 0.)`, `[0., 1.)`,
#' `[1., 2.)`, and `[2., +inf)`. For example, if the inputs are `boundaries` = [0, 10, 100]
#' input tensor = [[-5, 10000] [150, 10] [5, 100]]
#' then the output will be output = [[0, 3] [3, 2] [1, 3]]
#' ```
#' 
#' @param source_column A one-dimensional dense column which is generated with `numeric_column`.
#' @param boundaries A sorted list or list of floats specifying the boundaries.
#' 
#' @return A `_BucketizedColumn`.
#' 
#' @section Raises:
#' ValueError: If `source_column` is not a numeric column, or if it is not one-dimensional. 
#' ValueError: If `boundaries` is not a sorted list or list.
#'   
#' @family feature_column wrappers
#'   
#' @export
column_bucketized <- function(source_column, boundaries) {
  feature_column_lib$bucketized_column(
    source_column = source_column,
    boundaries = boundaries
  )
}


# pull the columns out of a tfestimators_feature_columns object
resolve_feature_columns <- function(feature_columns) {
  feature_columns$features
}


