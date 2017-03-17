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
  
  # path to package provided python modules
  python_path <- system.file("python", package = "tflearn")
  
  # add it to sys.path if it isn't already there
  sys <- import("sys", convert = FALSE)
  if (!python_path %in% reticulate::py_to_r(sys$path))
    sys$path$append(python_path)
  
  # import
  import(module, convert = convert)
}


