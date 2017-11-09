#' Get the Latest Checkpoint in a Checkpoint Directory
#' 
#' @param checkpoint_dir The path to the checkpoint directory.
#' @param ... Optional arguments passed on to \code{latest_checkpoint()}.
#' 
#' @export
#' @family utility functions
latest_checkpoint <- function(checkpoint_dir, ...) {
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

mv_tf_events_file <- function(model_dir) {
  tf_events_file_path <- file.path(model_dir, list.files(model_dir, pattern = "tfevents"))
  destination_path <- file.path(model_dir, "logs")
  dir.create(destination_path, showWarnings = FALSE)
  invisible(file.rename(from = tf_events_file_path, to = file.path(destination_path, basename(tf_events_file_path))))
}

# predict() expects at least "predictions" for predict_keys argument
resolve_predict_keys <- function(predict_keys) {
  if (length(predict_keys) == length(names(prediction_keys()))) {
    # preserve the default behavior of Python API
    NULL
  } else {
    predict_keys <- unlist(predict_keys)
    predictions_key <- prediction_keys()$PREDICTIONS
    if (!predictions_key %in% predict_keys)
      c(predict_keys, predictions_key)
    else
      predict_keys
  }
}

# if the model_dir is unspecified and there is a run_dir() available then 
# use the run_dir()
resolve_model_dir <- function(model_dir) {
  if (is.null(model_dir) && tfruns::is_run_active())
    tfruns::run_dir()
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

is.built_in_custom_hook <- function(hook) {
  is.list(hook) && identical(names(hook), c("hook_fn", "type"))
}

resolve_train_hooks <- function(hooks, steps, estimator) {
  
  .globals$history[[mode_keys()$TRAIN]] <- tf_estimator_history()
  
  hooks <- lapply(normalize_session_run_hooks(hooks), function(hook) {
    if (is.built_in_custom_hook(hook)) {
      type <- hook$type
      hook_fn <- hook$hook_fn
      if (type == "hook_history_saver") {
        hook_fn(mode_keys()$TRAIN)
      } else if (type == "hook_progress_bar") {
        hook_fn("Training", steps)
      }
    } else {
      hook
    }
  })

  normalize_session_run_hooks(hooks)
}

resolve_eval_hooks <- function(hooks, steps) {
  
  .globals$history[[mode_keys()$EVAL]] <- tf_estimator_history()

  hooks <- lapply(normalize_session_run_hooks(hooks), function(hook) {
    if (is.built_in_custom_hook(hook)) {
      type <- hook$type
      hook_fn <- hook$hook_fn
      if (type == "hook_history_saver") {
        hook_fn(mode_keys()$EVAL)
      } else if (type == "hook_progress_bar") {
        hook_fn("Evaluating", steps)
      }
    } else {
      hook
    }
  })
  
  normalize_session_run_hooks(hooks)
}
