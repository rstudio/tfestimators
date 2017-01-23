tf_backwards_compatibility_api <- function(envir = parent.frame()) {

  # retrieve dots
  dots <- eval(quote(list(...)), envir = envir)

  # if tf.options is NULL in envir, initialize it
  if (is.null(envir[["tf.options"]]))
    assign("tf.options", tf_options(), envir = envir)

  # if 'x' is a formula, and 'data' exists, then update
  # 'response' and 'x' in parent frame as appropriate
  if (is.formula(envir[["x"]]) && !is.null(dots[["data"]])) {
    assign("response", envir[["x"]], envir = envir)
    assign("x", dots[["data"]], envir = envir)
  }
}

tf_prepare_response_features_intercept <- function(x,
                                                   response,
                                                   features,
                                                   intercept,
                                                   envir = parent.frame())
{
  # if 'x' is a formula, and the 'data' argument has been supplied,
  # respect that

  # extract response from parent frame
  if (is.formula(response)) {
    parsed <- parse_formula(response, data = x)
    response <- parsed$response
    features <- parsed$features
    if (is.logical(parsed$intercept)) intercept <- parsed$intercept
  }

  # ensure types
  response <- ensure_scalar_character(response)
  features <- as.character(features)
  intercept <- ensure_scalar_boolean(intercept)

  # mutate in environment
  assign("response", response, envir = envir)
  assign("features", features, envir = envir)
  assign("intercept", intercept, envir = envir)

  x
}

# Construct column placeholders from vectors in an R object
tf_columns <- function(x, columns) {
  layers <- tf$contrib$layers
  lapply(columns, function(column) {
    v <- x[[column]]
    if (is.numeric(v)) {
      layers$real_valued_column(column)
    } else if (is.factor(v)) {
      layers$sparse_column_with_hash_bucket(column)
    } else if (is.character(v)) {
      layers$sparse_column_with_keys(column, keys = levels(v))
    }
  })
}

tf_setting <- function(name, default) {

  # Check for environment variable with associated name
  env <- toupper(gsub(".", "_", name, fixed = TRUE))
  val <- Sys.getenv(env, unset = NA)
  if (!is.na(val))
    return(val)

  # Check for R option with associated name
  val <- getOption(name)
  if (!is.null(val))
    return(val)

  # Use default value
  default
}
