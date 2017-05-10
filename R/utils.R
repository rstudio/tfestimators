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
  python_path <- system.file("python", package = "tfestimators")
  
  # add it to sys.path if it isn't already there
  sys <- import("sys", convert = FALSE)
  if (!python_path %in% reticulate::py_to_r(sys$path))
    sys$path$append(python_path)
  
  # import
  import(module, convert = convert)
}

#' Function to get the latest checkpoint in a checkpoint directory
#' 
#' @export
#' @family utility functions
get_latest_checkpoint <- function(checkpoint_dir, ...) {
  if (!dir.exists(checkpoint_dir)) {
    stop(paste0("This checkpoint_dir does not exist: ", checkpoint_dir))
  }
  tf$python$training$saver$latest_checkpoint(checkpoint_dir, ...) 
}


