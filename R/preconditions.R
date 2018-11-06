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
