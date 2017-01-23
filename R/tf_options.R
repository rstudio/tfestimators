#' Options for TensorFlow Routines
#'
#' Used to control the various facets of layers constructed with the
#' modeling routines included in this package.
#'
#' @param optimizer The optimizer to be used. When \code{NULL}, the default
#'   optimizer associated with the active estimator will be used.
#' @param steps The number of steps to be used when running the associated model.
#' @param model.dir The location where model outputs should be written. Defaults
#'   to a temporary directory within the \code{R} \code{\link{tempdir}}(), as
#'   produced by \code{\link{tempfile}}().
#' @export
tf_options <- function(
  optimizer = NULL,
  steps = 100L,
  model.dir = tf_setting("tf.model.dir", tempfile()))
{
  options <- list(
    optimizer = optimizer,
    steps = ensure_scalar_integer(steps),
    model.dir = model.dir
  )
  
  class(options) <- "tf_options"
  options
}
