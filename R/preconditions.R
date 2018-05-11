ensure_valid_column_names <- function(x, columns) {
  existed_cols <- object_names(x)
  invalid_columns <- !(columns %in% existed_cols)
  if (any(invalid_columns)) {
    stop(paste0("The following columns are not in the dataset: ",
                paste(columns[invalid_columns], collapse = ",")))
  }
}

object_names <- function(x) {
  
  # return character vectors as-is
  if (is.character(x))
    return(x)
  
  # if the object has column names, use those
  if (!is.null(colnames(x)))
    return(colnames(x))
  
  # otherwise, default to names
  names(x)
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

ensure_dict <- function(x, named = FALSE) {
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

ensure_nullable_list <- function(x) {
  result <- if (!is.null(x) && !is.list(x))
    list(x)
  else
    x
  unname(result)
}

as_nullable_integer <- function(x) {
  if (is.null(x))
    x
  else
    as.integer(x)
}

make_ensure_scalar_impl <- function(checker, message, converter) {
  fn <- function(object,
                 allow.na = FALSE,
                 allow.null = FALSE,
                 default = NULL)
  {
    object <- object %||% default
    
    if (allow.null && is.null(object)) return(object)

    if (!checker(object))
      stopf("'%s' is not %s", deparse(substitute(object)), message)

    if (is.na(object)) object <- NA_integer_
    if (!allow.na)     ensure_not_na(object)

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

ensure_scalar_integer <- function(x, allow.null = FALSE) {
  if (rlang::is_null(x) && allow.null)
    return(x)
  if (!rlang::is_bare_numeric(x, 1))
    stop(x, " is not a length-one numeric or integer vector",
         call. = FALSE)
  rlang::as_integer(x)
}

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
