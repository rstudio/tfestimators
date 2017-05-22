#' @export
train <- function(object, ...) {
  invisible(UseMethod("train"))
}

#' @export
evaluate <- function(object, ...) {
  UseMethod("evaluate")
}

#' @export
train_and_evaluate <- function(object, ...) {
  UseMethod("train_and_evaluate")
}

#' @export
export_savedmodel <- function(object, ...) {
  invisible(UseMethod("export_savedmodel"))
}

#' @export
experiment <- function(x, ...) {
  UseMethod("experiment")
}


