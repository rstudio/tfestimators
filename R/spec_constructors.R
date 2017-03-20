#' @export
feature_columns <- function(x, ...) {
  UseMethod("feature_columns")
}

#' Construct column placeholders from vectors in an R object
#' @export
feature_columns.default <- function(x, columns) {
  ensure_valid_column_names(x, columns)
  function() {
    lapply(columns, function(column) {
      v <- x[[column]]
      if (is.numeric(v)) {
        column_real_valued(column)
      } else if (is.factor(v)) {
        column_with_keys(column, keys = levels(v))
      } else if (is.character(v)) {
        column_with_hash_bucket(column)
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
  validate_input_fn_args(x, features, response)
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

validate_input_fn_args <- function(x, features, response) {
  ensure_valid_column_names(x, features)
  if (!is.null(response)) {
    ensure_valid_column_names(x, response)
  }
  force(list(x, features, response))
}
