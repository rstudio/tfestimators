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

check_dtype <- function(dtype) {
  if (!inherits(dtype, "tensorflow.python.framework.dtypes.DType")) {
    stop("dtype must of tf$DType objects, e.g. tf$int64")
  }
  dtype
}

is.tensor <- function(object) {
  inherits(object, "tensorflow.python.framework.ops.Tensor")
}


#' Model directory
#' 
#' Get the directory where a model's artifacts are stored.
#' 
#' @param object Model object
#' @param ... Unused
#'
#' @export
model_dir <- function(object, ...) {
  UseMethod("model_dir")
}


#' @export
model_dir.tf_estimator <- function(object, ...) {
  object$estimator$model_dir
}


# if the model_dir is unspecified and there is a run_dir() available then 
# use the run_dir()
resolve_model_dir <- function(model_dir) {
  if (is.null(model_dir) && !is.null(run_dir()))
    run_dir()
  else
    model_dir
}
