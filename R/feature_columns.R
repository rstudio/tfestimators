#' Feature Columns
#'
#' Constructors for feature columns. A feature column defines the expected
#' 'shape' of an input Tensor.
#'
#' @param ... One or more feature column definitions. The [tidyselect] package
#' is used to power generation of feature columns.
#' @param names Available feature names (for selection / pattern matching) as a
#'   character vector (or R object that implements `names()` or `colnames()`).
#'
#' @export
feature_columns <- function(..., names = NULL) {

  # scope names when they are provided
  if (!is.null(names))
    tidyselect::scoped_vars(object_names(names))
  
  # evaluate in an environment where 'tidyselect' is available
  data <- tidyselect_data()
  selections <- lapply(quos(...), function(quo) {
    
    # if this is a two-sided formula, transform call from e.g.
    #
    #     starts_with("a") ~ column_numeric()
    #
    # to
    #
    #     column_numeric(starts_with("a"))
    #
    expr <- quo_expr(quo)
    if (is_formula(expr, lhs = TRUE)) {
      
      # extract formula expressions
      lhs <- expr[[2]]; rhs <- expr[[3]]
      
      # inject lhs as first argument to rhs
      injected <- as.call(c(lang_head(rhs), lhs, lang_tail(rhs)))
      
      # update expression
      quo <- set_expr(quo, injected)
    }
    
    rlang::eval_tidy(quo, data = data)
  })
  
  # flatten our (potentially recursive) list
  flattened <- c(selections, recursive = TRUE)
  
  # remove duplicated columns (in case a column was
  # matched by multiple criterion
  keys <- vapply(flattened, function(item) {
    item$key
  }, character(1))
  dupes <- duplicated(keys)
  flattened[!dupes]
}

# Base documentation for feature column constructors ----

#' @param ... Expression(s) identifying input feature(s). Used as the column
#'   name and the dictionary key for feature parsing configs, feature tensors,
#'   and feature columns.
#'
#' @name column_base
NULL

#' Construct a Categorical Column with In-Memory Vocabulary
#'
#' Use this when your inputs are in string or integer format, and you have an
#' in-memory vocabulary mapping each value to an integer ID. By default,
#' out-of-vocabulary values are ignored. Use `default_value` to specify how to
#' include out-of-vocabulary values. For the input dictionary `features`,
#' `features$key` is either tensor or sparse tensor object. If it's tensor object,
#' missing values can be represented by `-1` for int and `''` for string.
#'
#' Note that these values are independent of the `default_value` argument.
#'
#' @inheritParams column_base
#'
#' @param vocabulary_list An ordered iterable defining the vocabulary. Each
#'   feature is mapped to the index of its value (if present) in
#'   `vocabulary_list`. Must be castable to `dtype`.
#' @param dtype The type of features. Only string and integer types are
#'   supported. If `NULL`, it will be inferred from `vocabulary_list`.
#' @param default_value The value to use for values not in `vocabulary_list`.
#' @param num_oov_buckets Non-negative integer, the number of out-of-vocabulary
#'   buckets. All out-of-vocabulary inputs will be assigned IDs in the range
#'   `[vocabulary_size, vocabulary_size+num_oov_buckets)` based on a hash of the
#'   input value. A positive `num_oov_buckets` can not be specified with
#'   `default_value`.
#' @return A categorical column with in-memory vocabulary.
#'
#' @section Raises: 
#' * ValueError: if `vocabulary_list` is empty, or contains
#' duplicate keys. 
#' * ValueError: if `dtype` is not integer or string.
#'
#' @export
#' @family feature column constructors
column_categorical_with_vocabulary_list <- function(...,
                                                    vocabulary_list,
                                                    dtype = NULL,
                                                    default_value = -1L,
                                                    num_oov_buckets = 0L)
{
  create_columns(..., f = function(column) {
    feature_column_lib$categorical_column_with_vocabulary_list(
      key = column,
      vocabulary_list = vocabulary_list,
      dtype = dtype,
      default_value = as_nullable_integer(default_value),
      num_oov_buckets = as.integer(num_oov_buckets)
    )
  })
}

#' Construct a Categorical Column with a Vocabulary File
#'
#' Use this when your inputs are in string or integer format, and you have a
#' vocabulary file that maps each value to an integer ID. By default,
#' out-of-vocabulary values are ignored. Use either (but not both) of
#' `num_oov_buckets` and `default_value` to specify how to include
#' out-of-vocabulary values. For input dictionary `features`, `features[key]` is
#' either tensor or sparse tensor object. If it's tensor object, missing values can be
#' represented by `-1` for int and `''` for string. Note that these values are
#' independent of the `default_value` argument.
#'
#' @inheritParams column_base
#'
#' @param vocabulary_file The vocabulary file name.
#' @param vocabulary_size Number of the elements in the vocabulary. This must be
#'   no greater than length of `vocabulary_file`, if less than length, later
#'   values are ignored.
#' @param num_oov_buckets Non-negative integer, the number of out-of-vocabulary
#'   buckets. All out-of-vocabulary inputs will be assigned IDs in the range
#'   `[vocabulary_size, vocabulary_size+num_oov_buckets)` based on a hash of the
#'   input value. A positive `num_oov_buckets` can not be specified with
#'   `default_value`.
#' @param default_value The integer ID value to return for out-of-vocabulary
#'   feature values, defaults to `-1`. This can not be specified with a positive
#'   `num_oov_buckets`.
#' @param dtype The type of features. Only string and integer types are
#'   supported.
#'
#' @return A categorical column with a vocabulary file.
#'
#' @section Raises:
#' * ValueError: `vocabulary_file` is missing.
#' * ValueError: `vocabulary_size` is missing or < 1.
#' * ValueError: `num_oov_buckets` is not a non-negative integer.
#' * ValueError: `dtype` is neither string nor integer.
#'
#' @family feature column constructors
#' @export
column_categorical_with_vocabulary_file <- function(...,
                                                    vocabulary_file,
                                                    vocabulary_size,
                                                    num_oov_buckets = 0L,
                                                    default_value = NULL,
                                                    dtype = tf$string)
{
  create_columns(..., f = function(column) {
    feature_column_lib$categorical_column_with_vocabulary_file(
      key = column,
      vocabulary_file = vocabulary_file,
      vocabulary_size = vocabulary_size,
      num_oov_buckets = as.integer(num_oov_buckets),
      default_value = as_nullable_integer(default_value),
      dtype = dtype
    )
  })
}

#' Construct a Categorical Column that Returns Identity Values
#'
#' Use this when your inputs are integers in the range `[0, num_buckets)`, and
#' you want to use the input value itself as the categorical ID. Values outside
#' this range will result in `default_value` if specified, otherwise it will
#' fail.
#'
#' Typically, this is used for contiguous ranges of integer indexes, but it
#' doesn't have to be. This might be inefficient, however, if many of IDs are
#' unused. Consider `categorical_column_with_hash_bucket` in that case.
#'
#' For input dictionary `features`, `features$key` is either tensor or sparse 
#' tensor object. If it's tensor object, missing values can be represented by `-1` for
#' int and `''` for string. Note that these values are independent of the
#' `default_value` argument.
#'
#' @inheritParams column_base
#'
#' @param num_buckets Number of unique values.
#' @param default_value If `NULL`, this column's graph operations will fail for
#'   out-of-range inputs. Otherwise, this value must be in the range `[0,
#'   num_buckets)`, and will replace inputs in that range.
#'
#' @return A categorical column that returns identity values.
#'
#' @section Raises: 
#' * ValueError: if `num_buckets` is less than one. 
#' * ValueError: if `default_value` is not in range `[0, num_buckets)`.
#'
#' @family feature column constructors
#' @export
column_categorical_with_identity <- function(...,
                                             num_buckets,
                                             default_value = NULL)
{
  create_columns(..., f = function(column) {
    feature_column_lib$categorical_column_with_identity(
      key = column,
      num_buckets = as.integer(num_buckets),
      default_value = as_nullable_integer(default_value)
    )

  })
}

#' Represents Multi-Hot Representation of Given Categorical Column
#'
#' Used to wrap any `categorical_column_*` (e.g., to feed to DNN). Use
#' `embedding_column` if the inputs are sparse.
#'
#' @param categorical_column A categorical column which is created by
#'   the `categorical_column_with_*()` or `crossed_column()` functions.
#'
#' @return An indicator column.
#'
#' @export
column_indicator <- function(categorical_column) {
  feature_column_lib$indicator_column(
    categorical_column = categorical_column
  )
}

#' Represents Sparse Feature where IDs are set by Hashing
#'
#' Use this when your sparse features are in string or integer format, and you
#' want to distribute your inputs into a finite number of buckets by hashing.
#' output_id = Hash(input_feature_string) % bucket_size For input dictionary
#' `features`, `features$key$` is either tensor or sparse tensor object. If it's 
#' tensor object, missing values can be represented by `-1` for int and `''` for
#' string. Note that these values are independent of the `default_value`
#' argument.
#'
#' @inheritParams column_base
#'
#' @param hash_bucket_size An int > 1. The number of buckets.
#' @param dtype The type of features. Only string and integer types are
#'   supported.
#'
#' @return A `_HashedCategoricalColumn`.
#'
#' @section Raises: 
#' * ValueError: `hash_bucket_size` is not greater than 1.
#' * ValueError: `dtype` is neither string nor integer.
#'
#' @family feature column constructors
#'
#' @export
column_categorical_with_hash_bucket <- function(...,
                                                hash_bucket_size,
                                                dtype = tf$string)
{
  hash_bucket_size <- as.integer(hash_bucket_size)
  if (hash_bucket_size <= 1) {
    stop("hash_bucket_size must be larger than 1")
  }

  create_columns(..., f = function(column) {
    feature_column_lib$categorical_column_with_hash_bucket(
      key = column,
      hash_bucket_size = hash_bucket_size,
      dtype = dtype
    )
  })
}

#' Construct a Real-Valued Column
#'
#' @inheritParams column_base
#'
#' @param shape An integer vector that specifies the shape of the tensor. An
#'   integer can be given which means a single dimension tensor with given
#'   width. The tensor representing the column will have the shape of
#'   `batch_size + shape`.
#' @param default_value A single value compatible with `dtype` or an iterable of
#'   values compatible with `dtype` which the column takes on during parsing if
#'   data is missing. A default value of `NULL` will cause `tf$parse_example` to
#'   fail if an example does not contain this column. If a single value is
#'   provided, the same value will be applied as the default value for every
#'   item. If an iterable of values is provided, the shape of the
#'   `default_value` should be equal to the given `shape`.
#' @param dtype The types for values contained in the column. The default value
#'   is `tf$float32`. Must be a non-quantized, real integer or floating point
#'   type.
#' @param normalizer_fn If not `NULL`, a function that can be used to normalize
#'   the value of the tensor after `default_value` is applied for parsing.
#'   Normalizer function takes the input `Tensor` as its argument, and returns
#'   the output tensor. (e.g. `function(x) {(x - 3.0) / 4.2)}`. Please note that
#'   even though the most common use case of this function is normalization, it
#'   can be used for any kind of Tensorflow transformations.
#'
#' @return A numeric column.
#'
#' @section Raises: 
#' * TypeError: if any dimension in shape is not an int
#' * ValueError: if any dimension in shape is not a positive integer 
#' * TypeError: if `default_value` is an iterable but not compatible with `shape`
#' * TypeError: if `default_value` is not compatible with `dtype`
#' * ValueError: if `dtype` is not convertible to `tf$float32`
#'
#' @family feature column constructors
#'
#' @export
column_numeric <- function(...,
                           shape = c(1L),
                           default_value = NULL,
                           dtype = tf$float32,
                           normalizer_fn = NULL)
{
  create_columns(..., f = function(column) {
    feature_column_lib$numeric_column(
      key = column,
      shape = as.integer(shape),
      default_value = default_value,
      dtype = dtype,
      normalizer_fn = normalizer_fn
    )
  })
}

#' Construct a Dense Column
#'
#' Use this when your inputs are sparse, but you want to convert them to a dense
#' representation (e.g., to feed to a DNN). Inputs must be a
#' categorical column created by any of the `column_categorical_*()`
#' functions.
#'
#' @param categorical_column A categorical column created by a
#'   `column_categorical_*()` function. This column produces the sparse IDs
#'   that are inputs to the embedding lookup.
#' @param dimension A positive integer, specifying dimension of the embedding.
#' @param combiner A string specifying how to reduce if there are multiple
#'   entries in a single row. Currently `"mean"`, `"sqrtn"` and `"sum"` are
#'   supported, with `"mean"` the default. `"sqrtn"`' often achieves good
#'   accuracy, in particular with bag-of-words columns. Each of this can be
#'   thought as example level normalizations on the column.
#' @param initializer A variable initializer function to be used in embedding
#'   variable initialization. If not specified, defaults to
#'   `tf$truncated_normal_initializer` with mean `0.0` and standard deviation
#'   `1 / sqrt(dimension)`.
#' @param ckpt_to_load_from String representing checkpoint name/pattern from
#'   which to restore column weights. Required if `tensor_name_in_ckpt` is not
#'   `NULL`.
#' @param tensor_name_in_ckpt Name of the `Tensor` in `ckpt_to_load_from` from
#'   which to restore the column weights. Required if `ckpt_to_load_from` is not
#'   `NULL`.
#' @param max_norm If not `NULL`, embedding values are l2-normalized to this
#'   value.
#' @param trainable Whether or not the embedding is trainable. Default is TRUE.
#'
#' @return A dense column that converts from sparse input.
#'
#' @section Raises: 
#' * ValueError: if `dimension` not > 0. 
#' * ValueError: if exactly one of `ckpt_to_load_from` and `tensor_name_in_ckpt`
#' is specified.
#' * ValueError: if `initializer` is specified and is not callable.
#'
#' @family feature column constructors
#'
#' @export
column_embedding <- function(categorical_column,
                             dimension,
                             combiner = "mean",
                             initializer = NULL,
                             ckpt_to_load_from = NULL,
                             tensor_name_in_ckpt = NULL,
                             max_norm = NULL,
                             trainable = TRUE)
{
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

#' Construct a Crossed Column
#'
#' Returns a column for performing crosses of categorical features. Crossed
#' features will be hashed according to `hash_bucket_size`.
#'
#' @param keys An iterable identifying the features to be crossed. Each element
#'   can be either: 
#'   * string: Will use the corresponding feature which must be of string type. 
#'   * categorical column: Will use the transformed tensor
#'   produced by this column. Does not support hashed categorical columns.
#' @param hash_bucket_size The number of buckets (> 1).
#' @param hash_key Optional: specify the hash_key that will be used by the
#'   `FingerprintCat64` function to combine the crosses fingerprints on
#'   `SparseCrossOp`.
#'
#' @return A crossed column.
#'
#' @section Raises: 
#' * ValueError: If `len(keys) < 2`. 
#' * ValueError: If any of the keys is neither a string nor categorical column. 
#' * ValueError: If any of the keys is `_HashedCategoricalColumn`. 
#' * ValueError: If `hash_bucket_size < 1`.
#'
#' @family feature column constructors
#'
#' @export
column_crossed <- function(keys,
                           hash_bucket_size,
                           hash_key = NULL)
{
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


#' Construct a Weighted Categorical Column
#'
#' Use this when each of your sparse inputs has both an ID and a value. For
#' example, if you're representing text documents as a collection of word
#' frequencies, you can provide 2 parallel sparse input features ('terms' and
#' 'frequencies' below).
#'
#' @param categorical_column A categorical column created by
#'   `column_categorical_*()` functions.
#' @param weight_feature_key String key for weight values.
#' @param dtype Type of weights, such as `tf$float32`. Only float and integer
#'   weights are supported.
#'
#' @return A categorical column composed of two sparse features: one
#'   represents id, the other represents weight (value) of the id feature in
#'   that example.
#'
#' @section Raises: 
#' * ValueError: if `dtype` is not convertible to float.
#'
#' @family feature column constructors
#'
#' @export
column_categorical_weighted <- function(categorical_column,
                                        weight_feature_key,
                                        dtype = tf$float32)
{
  feature_column_lib$weighted_categorical_column(
    categorical_column = categorical_column,
    weight_feature_key = weight_feature_key,
    dtype = dtype
  )
}


#' Construct a Bucketized Column
#'
#' Construct a bucketized column, representing discretized dense input. Buckets
#' include the left boundary, and exclude the right boundary.
#'
#' @param source_column A one-dimensional dense column, as generated by [column_numeric()].
#' @param boundaries A sorted list or list of floats specifying the boundaries.
#'
#' @return A bucketized column.
#'
#' @section Raises:
#' * ValueError: If `source_column` is not a numeric column, or if it is not one-dimensional.
#' * ValueError: If `boundaries` is not a sorted list or list.
#'
#' @family feature column constructors
#'
#' @export
column_bucketized <- function(source_column, boundaries) {
  feature_column_lib$bucketized_column(
    source_column = source_column,
    boundaries = boundaries
  )
}

#' Construct an Input Layer
#'
#' Returns a dense tensor as input layer based on given `feature_columns`.
#' At the first layer of the model, this column oriented data should be converted
#' to a single tensor.
#'
#' @param features A mapping from key to tensors. Feature columns look up via
#'   these keys. For example `column_numeric('price')` will look at 'price' key
#'   in this dict. Values can be a sparse tensor or tensor depends on
#'   corresponding feature column.
#' @param feature_columns An iterable containing the FeatureColumns to use as
#'   inputs to your model. All items should be instances of classes derived from
#'   a dense column such as [column_numeric()], [column_embedding()],
#'   [column_bucketized()], [column_indicator()]. If you have categorical features,
#'   you can wrap them with an [column_embedding()] or [column_indicator()].
#' @param weight_collections A list of collection names to which the Variable
#'   will be added. Note that, variables will also be added to collections
#'   `graph_keys()$GLOBAL_VARIABLES` and `graph_keys()$MODEL_VARIABLES`.
#' @param trainable If `TRUE` also add the variable to the graph collection
#'   `graph_keys()$TRAINABLE_VARIABLES` (see `tf$Variable`).
#'
#' @return A tensor which represents input layer of a model. Its shape is
#'   (batch_size, first_layer_dimension) and its dtype is `float32`.
#'   first_layer_dimension is determined based on given `feature_columns`.
#'
#' @section Raises: 
#' * ValueError: if an item in `feature_columns` is not a dense column.
#'
#' @family feature column constructors
#' @export
input_layer <- function(features,
                        feature_columns,
                        weight_collections = NULL,
                        trainable = TRUE)
{
  tf$feature_column$input_layer(
    features = features,
    feature_columns = feature_columns,
    weight_collections = weight_collections,
    trainable = trainable
  )
}

create_columns <- function(..., f) {

  columns <- tryCatch(
    names(vars_select(peek_vars(), !!! quos(...))),
    error = function(e) {
      as.character(c(...))
    }
  ) 

  columns <- lapply(columns, f)
  if (length(columns) == 1)
    columns[[1]]
  else
    columns
}
