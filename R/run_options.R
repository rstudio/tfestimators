#' Options for TensorFlow Routines
#'
#' Used to control the various facets of layers constructed with the
#' modeling routines included in this package.
#'
#' @param steps The number of steps to be used when running the associated model.
#' @param model.dir The location where model outputs should be written. Defaults
#'   to a temporary directory within the \code{R} \code{\link{tempdir}}(), as
#'   produced by \code{\link{tempfile}}().
#' @export
run_options <- function(
  steps = 30L,
  model.dir = tf_setting("tf.model.dir", tempfile("tflearn_")),
  run.config = learn$RunConfig(tf_random_seed=1))
{
  options <- list(
    steps = ensure_scalar_integer(steps),
    model.dir = model.dir,
    run.config = run.config
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
