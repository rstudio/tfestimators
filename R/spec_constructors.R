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

#' @export
input_fn <- function(x, ...) {
  UseMethod("input_fn")
}

#' @export
input_fn.formula <- function(x, data, ...) {
  parsed <- parse_formula(x)
  input_fn(data, parsed$features, parsed$response, ...) # TODO: Support unsupervised algorithms
}

#' @export
input_fn.default <-  function(
  x,
  features,
  response = NULL)
{
  validate_input_fn_args(x, features, response)
  function(features_as_named_list) {
    function() {
      if (features_as_named_list) {
        # For canned estimators
        input_features <- lapply(features, function(feature) {
          if(is.factor(x[[feature]])) {
            # factor vars are incorrectly converted as Tensor of type int
            tf$constant(as.character(x[[feature]]))
          } else {
            tf$constant(x[[feature]])
          }
        })
        names(input_features) <- features
      } else {
        # For custom estimators
        input_features <- tf$constant(as.matrix(x[, features]))
      }
      if (!is.null(response)) {
        input_response <- tf$constant(x[[response]])
      } else {
        input_response <- NULL
      }
      list(input_features, input_response)
    }
  }
}

#' @export
input_fn.list <- function(
  x,
  features,
  response # TODO: Support unsupervised
) {
  validate_input_fn_args(x, features, response)
  function(features_as_named_list) {
    inputs <- tf$constant(
      np$array(
        x$features,
        dtype = np$int64
      )
    )
    labels <- tf$constant(
      np$array(
        x$response
      )
    )
    list(list(inputs = inputs), labels)
  }
}

#' @export
input_fn.data.frame <-  function(
  x,
  features,
  response = NULL,
  batch_size = 10L,
  shuffle = TRUE)
{
  validate_input_fn_args(x, features, response)
  # supporting for unsupervised models as well as ingesting data for inference
  input_response <- if(is.null(response)) NULL else as.array(x[,response])
  fn <- function(features_as_named_list) {
    if(features_as_named_list){
      values <- lapply(features, function(feature) {
        as.array(x[, feature])
      })
      names(values) <- features
      fn <- tf$estimator$inputs$numpy_input_fn(
        dict(values),
        input_response,
        batch_size = batch_size,
        shuffle = shuffle)
    } else {
      values <- list(features = data.matrix(x)[,features, drop = FALSE])
      fn <- function(){
        fun <- tf$estimator$inputs$numpy_input_fn(
          values,
          input_response,
          batch_size = batch_size,
          shuffle = shuffle)
        fun <- fun()
        list(
          fun[[1]]$features,
          fun[[2]]
        )
      }
    }
  }
}


#' @export
input_fn.matrix <- function(
  x,
  features,
  response = NULL,
  batch_size = 10L,
  shuffle = TRUE
) {
  input_fn(as.data.frame(x), features, response, batch_size, shuffle)
}

validate_input_fn <- function(input_fn) {
  if (is.null(input_fn$input_fn)) {
    stop("Your input_fn must return a list with one item: input_fn")
  }
  if (!is.function(input_fn$input_fn)) {
    stop("Your input_fn$input_fn must be a function")
  }
}

#' @export
validate_input_fn_args <- function(x, features, response) {
  ensure_valid_column_names(x, features)
  if (!is.null(response)) {
    ensure_valid_column_names(x, response)
  }
  force(list(x, features, response))
}
