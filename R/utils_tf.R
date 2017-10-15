#' Get the Latest Checkpoint in a Checkpoint Directory
#' 
#' @param checkpoint_dir The path to the checkpoint directory.
#' @param ... Optional arguments passed on to \code{latest_checkpoint()}.
#' 
#' @export
#' @family utility functions
get_latest_checkpoint <- function(checkpoint_dir, ...) {
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
  if (!is.null(predict_keys)) {
    predict_keys <- unlist(predict_keys)
    predictions_key <- prediction_keys()$PREDICTIONS
    if (!predictions_key %in% predict_keys)
      c(predict_keys, predictions_key)
    else
      predict_keys
  } else {
   NULL 
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

# determine whether to view metrics or not
resolve_view_metrics <- function(view_metrics, verbose) {

  if (identical(view_metrics, "auto"))
    view_metrics <- verbose
  
  # TODO: enable outside of RStudio?
  if (is.null(getOption("viewer")) || is.na(Sys.getenv("RSTUDIO", unset = NA)))
    view_metrics <- FALSE
  
  view_metrics
}

resolve_train_hooks <- function(hooks, verbose, steps, view_metrics, estimator) {
  if (verbose) {
    .globals$history <- tf_estimator_history()
    hooks <- c(hooks, hook_history_saver())
    hooks <- c(hooks, hook_progress_bar("Training", steps))
  }
  
  if (resolve_view_metrics(view_metrics, verbose))
    hooks <- c(
      hooks,
      hook_view_metrics(
        list(
          steps = steps,
          model = str(estimator)
        )
      )
    )
  
  normalize_session_run_hooks(hooks)
}

resolve_eval_hooks <- function(hooks, verbose, steps) {
  if (verbose) {
    .globals$history <- tf_estimator_history()
    hooks <- c(hooks, hook_history_saver())
    hooks <- c(hooks, hook_progress_bar("Evaluating", steps))
  }
  
  normalize_session_run_hooks(hooks)
}
