#' Options for TensorFlow Routines
#'
#' Used to control the various facets of layers constructed with the
#' modeling routines included in this package.
#'
#' @param model_dir The location where model outputs should be written. Defaults
#'   to a temporary directory within the \code{R} \code{\link{tempdir}}(), as
#'   produced by \code{\link{tempfile}}().
#' @param run_config A learn$RunConfig object, e.g. learn$RunConfig(tf_random_seed = 1)
#' that specifies the run-time configuration of a model operation.
#' @export
run_options <- function(
  model_dir  = tf_setting("tflearn.model_dir", tempfile("tflearn_")),
  run_config = learn$RunConfig(tf_random_seed = 1))
{
  options <- list(
    model_dir = model_dir,
    run_config = run_config
  )

  class(options) <- "run_options"
  options
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
