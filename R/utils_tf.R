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
