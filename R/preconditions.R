#' @export
ensure_valid_column_names <- function(x, columns) {
  existed_cols <- colnames(x)
  invalid_columns <- ! columns %in% existed_cols
  if (any(invalid_columns)) {
    stop(paste0("The following columns are not in the dataset: ",
                paste(columns[invalid_columns], collapse = ",")))
  }
}

ensure_not_na <- function(object) {
  if (any(is.na(object))) {
    stopf(
      "'%s' %s",
      deparse(substitute(object)),
      if (length(object) > 1) "contains NA values" else "is NA"
    )
  }

  object
}

ensure_not_null <- function(object) {
  object %||% stop(sprintf("'%s' is NULL", deparse(substitute(object))))
}

ensure_scalar <- function(object) {

  if (length(object) != 1 || !is.numeric(object)) {
    stopf(
      "'%s' is not a length-one numeric value",
      deparse(substitute(object))
    )
  }
  object
}

ensure_dict <- function(x, named = F) {
  if (is.list(x)) {
    if (named && is.null(names(x))) {
      stop("x must be a named list")
    }
    dict(x)
  } else if (inherits(x, "python.builtin.dict")) {
    x
  } else {
    stop("x needs to be a list or dict")
  }
}

as_nullable_integer <- function(x) {
  if (is.null(x))
    x
  else
    as.integer(x)
}

check_dtype <- function(dtype) {
  if (!inherits(dtype, "tensorflow.python.framework.dtypes.DType")) {
    stop("dtype must of tf$DType objects, e.g. tf$int64")
  }
  dtype
}

make_ensure_scalar_impl <- function(checker, message, converter) {
  fn <- function(object,
                 allow.na = FALSE,
                 allow.null = FALSE,
                 default = NULL)
  {
    object <- object %||% default

    if (!checker(object))
      stopf("'%s' is not %s", deparse(substitute(object)), message)

    if (is.na(object)) object <- NA_integer_
    if (!allow.na)     ensure_not_na(object)
    if (!allow.null)   ensure_not_null(object)

    converter(object)
  }

  environment(fn) <- parent.frame()

  body(fn) <- do.call(
    substitute,
    list(
      body(fn),
      list(
        checker = substitute(checker),
        message = substitute(message),
        converter = substitute(converter)
      )
    )
  )

  fn
}

ensure_scalar_integer <- make_ensure_scalar_impl(
  is.numeric,
  "a length-one integer vector",
  as.integer
)

ensure_scalar_double <- make_ensure_scalar_impl(
  is.numeric,
  "a length-one numeric vector",
  as.double
)

ensure_scalar_boolean <- make_ensure_scalar_impl(
  is.logical,
  "a length-one logical vector",
  as.logical
)

ensure_scalar_character <- make_ensure_scalar_impl(
  is.character,
  "a length-one character vector",
  as.character
)


require_file_exists <- function(path, fmt = NULL) {
  fmt <- fmt %||% "no file at path '%s'"
  if (!file.exists(path)) stopf(fmt, path)
  path
}

require_directory_exists <- function(path, fmt = NULL) {
  fmt <- fmt %||% "no file at path '%s'"
  require_file_exists(path)
  info <- file.info(path)
  if (!isTRUE(info$isdir)) stopf(fmt, path)
  path
}

ensure_directory <- function(path) {

  if (file.exists(path)) {
    info <- file.info(path)
    if (isTRUE(info$isdir)) return(path)
    stopf("path '%s' exists but is not a directory", path)
  }

  if (!dir.create(path, recursive = TRUE))
    stopf("failed to create directory at path '%s'", path)

  invisible(path)

}
