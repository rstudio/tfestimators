#' Function to get the latest checkpoint in a checkpoint directory
#' 
#' @param checkpoint_dir The path to the checkpoint directory.
#' @param ... Optional arguments passed on to \code{latest_checkpoint()}.
#' 
#' @export
#' @family utility functions
get_latest_checkpoint <- function(checkpoint_dir, ...) {
  if (!dir.exists(checkpoint_dir)) {
    stop(paste0("This checkpoint_dir does not exist: ", checkpoint_dir))
  }
  tf$python$training$saver$latest_checkpoint(checkpoint_dir, ...) 
}


list_variable_names <- function(model_dir) {
  lapply(list_variables(model_dir), function(var) var[[1]])
}

list_variable_shapes <- function(model_dir) {
  lapply(list_variables(model_dir), function(var) var[[2]])
}

list_variables <- function(model_dir) {
  tf$python$training$checkpoint_utils$list_variables(model_dir)
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

resolve_activation_fn <- function(activation_fn) {
  
  # resolve activation functions specified by name in 'tf$nn' module
  if (is.character(activation_fn) && length(activation_fn) == 1) {
    if (!activation_fn %in% names(tf$nn)) {
      fmt <- "'%s' is not a known activation function in the 'tf$nn' module"
      stopf(fmt, activation_fn)
    }
    activation_fn <- tf$nn[[activation_fn]]
  }
  
  activation_fn
}

#' Standard names to use for graph collections.
#' 
#' The standard library uses various well-known names to collect and retrieve 
#' values associated with a graph.
#' 
#' For example, the `tf$Optimizer` subclasses default to optimizing the 
#' variables collected under`graph_keys()$TRAINABLE_VARIABLES` if `NULL` is 
#' specified, but it is also possible to pass an explicit list of variables.
#' 
#' The following standard keys are defined:
#' 
#' * `GLOBAL_VARIABLES`: the default collection of `Variable` objects, shared 
#' across distributed environment (model variables are subset of these). See 
#' `tf$global_variables` for more details. Commonly, all `TRAINABLE_VARIABLES` 
#' variables will be in `MODEL_VARIABLES`, and all `MODEL_VARIABLES` variables 
#' will be in `GLOBAL_VARIABLES`.
#' 
#' * `LOCAL_VARIABLES`: the subset of `Variable` objects that are local to each 
#' machine. Usually used for temporarily variables, like counters. Note: use 
#' `tf.contrib.framework.local_variable` to add to this collection.
#' 
#' * `MODEL_VARIABLES`: the subset of `Variable` objects that are used in the 
#' model for inference (feed forward). Note: use 
#' `tf.contrib.framework.model_variable` to add to this collection.
#' 
#' * `TRAINABLE_VARIABLES`: the subset of `Variable` objects that will be
#' trained by an optimizer. See @{tf.trainable_variables} for more details.
#' 
#' * `SUMMARIES`: the summary `Tensor` objects that have been created in the 
#' graph. See @{tf.summary.merge_all} for more details.
#' 
#' * `QUEUE_RUNNERS`: the `QueueRunner` objects that are used to produce input
#' for a computation. See `tf$train$start_queue_runners` for more details.
#' 
#' * `MOVING_AVERAGE_VARIABLES`: the subset of `Variable` objects that will also
#' keep moving averages. See `tf$moving_average_variables` for more details.
#' 
#' * `REGULARIZATION_LOSSES`: regularization losses collected during graph 
#' construction. The following standard keys are defined, but their 
#' collections are **not** automatically populated as many of the others are:
#' 
#'   * `WEIGHTS` 
#'   * `BIASES` 
#'   * `ACTIVATIONS`
#' 
graph_keys <- function() {
  tf$python$framework$ops$GraphKeys()
}

print.tensorflow.python.framework.ops.GraphKeys <- function(object) {
  cat(paste0("Available graph keys: ", paste(names(graph_keys()), collapse = ", ")))
}
