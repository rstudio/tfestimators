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
  response = NULL,
  features_as_named_list = TRUE)
{
  validate_input_fn_args(x, features, response, features_as_named_list)
  fn <- function() {
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
  return(list(
    input_fn = fn,
    features_as_named_list = features_as_named_list))
}

validate_input_fn <- function(input_fn) {
  if (is.null(input_fn$input_fn) || is.null(input_fn$features_as_named_list)) {
    stop("Your input_fn must return a list with two items: input_fn and features_as_named_list")
  }
  if (!is.function(input_fn$input_fn)) {
    stop("Your input_fn$input_fn must be a function")
  }
  if (!is.logical(input_fn$features_as_named_list)) {
    stop("Your input_fn$features_as_named_list must be logical")
  }
}

validate_input_fn_args <- function(x, features, response, features_as_named_list) {
  ensure_valid_column_names(x, features)
  if (!is.null(response)) {
    ensure_valid_column_names(x, response)
  }
  if (!is.logical(features_as_named_list)) {
    stop("features_as_named_list must be logical")
  }
  force(list(x, features, response))
}
