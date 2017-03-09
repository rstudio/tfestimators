#' Construct column placeholders from vectors in an R object
#' @export
construct_feature_columns <- function(x, columns) {
  ensure_valid_column_names(x, columns)
  column_list <- lapply(columns, function(column) {
    v <- x[[column]]
    if (is.numeric(v)) {
      column_real_valued(column)
    } else if (is.factor(v)) {
      column_with_keys(column, keys = levels(v))
    } else if (is.character(v)) {
      column_with_hash_bucket(column)
    }
  })
  function(){column_list}
}

#' @export
construct_input_fn <-  function(x, response, features, feature_as_named_list = TRUE, id_column = NULL) {
  ensure_valid_column_names(x, features)
  ensure_valid_column_names(x, response)
  # TODO: Consider removing this part
  if (!is.null(id_column)) {
    x[id_column] <- as.character(1:nrow(x))
    features <- c(features, id_column)
    # TODO: Support custom id_column function
  }
  force(list(x, response, features))
  function() {
    if (feature_as_named_list) {
      # For linear and dnn we have to do this due to nature of feature columns
      feature_columns <- lapply(features, function(feature) {
        tf$constant(x[[feature]])
      })
      names(feature_columns) <- features
    } else {
      # This works for custom model
      # TODO: Consider a separate spec constructor
      feature_columns <- tf$constant(as.matrix(x[, features]))
    }
    response_column <- tf$constant(x[[response]])
    list(feature_columns, response_column)
  }
}