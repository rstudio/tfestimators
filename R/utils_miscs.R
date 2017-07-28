"%||%" <- function(x, y) if (is.null(x)) y else x

stopf <- function(fmt, ..., call. = TRUE, domain = NULL) {
  stop(simpleError(
    sprintf(fmt, ...),
    if (call.) sys.call(sys.parent())
  ))
}

warnf <- function(fmt, ..., call. = TRUE, immediate. = FALSE) {
  warning(sprintf(fmt, ...), call. = call., immediate. = immediate.)
}

enumerate <- function(object, f, ...) {
  nm <- names(object)
  result <- lapply(seq_along(object), function(i) {
    f(nm[[i]], object[[i]], ...)
  })
  names(result) <- names(object)
  result
}

# make sure an object is a function
resolve_fn <- function(object) {
  if (is.function(object))
    object()
  else
    object
}


# utility function for importing python modules defined in the
# inst/python directory of the package
import_package_module <- function(module, convert = TRUE) {
  python_path <- system.file("python", package = "tfestimators")
  import_from_path(module, python_path, convert = convert)
}

