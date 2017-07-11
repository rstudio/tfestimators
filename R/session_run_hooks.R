#' Prints the given tensors every N local steps, every N seconds, or at end.
#' 
#' The tensors will be printed to the log, with `INFO` severity.
#' 
#' Note that if `at_end` is True, `tensors` should not include any tensor 
#' whose evaluation produces a side effect such as consuming additional inputs.
#' 
#' @param tensors `dict` that maps string-valued tags to tensors/tensor names,
#'   or `iterable` of tensors/tensor names.
#' @param every_n_iter `int`, print the values of `tensors` once every N local
#'   steps taken on the current worker.
#' @param every_n_secs `int` or `float`, print the values of `tensors` once
#'   every N seconds. Exactly one of `every_n_iter` and `every_n_secs` should be
#'   provided.
#' @param formatter function, takes dict of `tag`->`Tensor` and returns a
#'   string. If `NULL` uses default printing all tensors.
#' @param at_end `bool` specifying whether to print the values of `tensors` at the
#' end of the run.
#'   
#' @family session_run_hook wrappers
#'   
#' @export
hook_logging_tensor <- function(tensors,
                                every_n_iter = NULL,
                                every_n_secs = NULL,
                                formatter = NULL,
                                at_end = FALSE)
{
  tf$python$training$basic_session_run_hooks$LoggingTensorHook(
    tensors = ensure_dict(tensors, named = F),
    every_n_iter = as_nullable_integer(every_n_iter),
    every_n_secs = as_nullable_integer(every_n_secs),
    formatter = formatter,
    at_end = at_end
  )
}

#' Monitor to request stop at a specified step.
#' 
#' @param num_steps Number of steps to execute.
#' @param last_step Step after which to stop.
#' 
#' @family session_run_hook wrappers
#' 
#' @export
hook_stop_at_step <- function(num_steps = NULL, last_step = NULL) {
  if (!is.null(num_steps) && !is.null(last_step)) {
    stop(" Only one of num_steps or last_step can be specified")
  }
  tf$python$training$basic_session_run_hooks$StopAtStepHook(
    num_steps = as_nullable_integer(num_steps),
    last_step = as_nullable_integer(last_step)
  )
}


#' Saves checkpoints every N steps or seconds.
#' 
#' @param checkpoint_dir `str`, base directory for the checkpoint files.
#' @param save_secs `int`, save every N secs.
#' @param save_steps `int`, save every N steps.
#' @param saver `Saver` object, used for saving.
#' @param checkpoint_basename `str`, base name for the checkpoint files.
#' @param scaffold `Scaffold`, use to get saver object.
#' @param listeners List of `CheckpointSaverListener` subclass instances. Used
#'   for callbacks that run immediately after the corresponding
#'   CheckpointSaverHook callbacks, only in steps where the CheckpointSaverHook
#'   was triggered.
#'   
#' @family session_run_hook wrappers
#'   
#' @export
hook_checkpoint_saver <- function(checkpoint_dir,
                                  save_secs = NULL,
                                  save_steps = NULL,
                                  saver = NULL,
                                  checkpoint_basename = "model.ckpt",
                                  scaffold = NULL,
                                  listeners = NULL)
{
  if (!is.null(save_secs) && !is.null(save_steps)) {
    stop(" Only one of save_secs or save_steps can be specified")
  }
  
  tf$python$training$basic_session_run_hooks$CheckpointSaverHook(
    checkpoint_dir = checkpoint_dir,
    save_secs = as_nullable_integer(save_secs),
    save_steps = as_nullable_integer(save_steps),
    saver = saver,
    checkpoint_basename = checkpoint_basename,
    scaffold = scaffold,
    listeners = listeners
  )
}


#' Steps per second monitor.
#' 
#' @param every_n_steps every_n_steps
#' @param every_n_secs every_n_secs
#' @param output_dir output_dir
#' @param summary_writer summary_writer
#' 
#' @family session_run_hook wrappers
#' 
#' @export
hook_step_counter <- function(every_n_steps = 100L, every_n_secs = NULL, output_dir = NULL, summary_writer = NULL) {
  tf$python$training$basic_session_run_hooks$StepCounterHook(
    every_n_steps = as.integer(every_n_steps),
    every_n_secs = as_nullable_integer(every_n_secs),
    output_dir = output_dir,
    summary_writer = summary_writer
  )
}

#' NaN Loss monitor.
#' 
#' Monitors loss and stops training if loss is NaN. Can either fail with
#' exception or just stop training.
#' 
#' @param loss_tensor `Tensor`, the loss tensor.
#' @param fail_on_nan_loss `bool`, whether to raise exception when loss is NaN.
#'   
#' @family session_run_hook wrappers
#'   
#' @export
hook_nan_tensor <- function(loss_tensor, fail_on_nan_loss = TRUE) {
  tf$python$training$basic_session_run_hooks$NanTensorHook(
    loss_tensor = loss_tensor,
    fail_on_nan_loss = fail_on_nan_loss
  )
}

#' Saves summaries every N steps.
#' 
#' 
#' 
#' @param save_steps `int`, save summaries every N steps. Exactly one of
#'   `save_secs` and `save_steps` should be set.
#' @param save_secs `int`, save summaries every N seconds.
#' @param output_dir `string`, the directory to save the summaries to. Only used
#'   if no `summary_writer` is supplied.
#' @param summary_writer `SummaryWriter`. If `NULL` and an `output_dir` was
#'   passed, one will be created accordingly.
#' @param scaffold `Scaffold` to get summary_op if it's not provided.
#' @param summary_op `Tensor` of type `string` containing the serialized
#'   `Summary` protocol buffer or a list of `Tensor`. They are most likely an
#'   output by TF summary methods like `tf.summary.scalar` or
#'   `tf.summary.merge_all`. It can be passed in as one tensor; if more than
#'   one, they must be passed in as a list.
#'   
#' @family session_run_hook wrappers
#'   
#' @export
hook_summary_saver <- function(save_steps = NULL, save_secs = NULL, output_dir = NULL, summary_writer = NULL, scaffold = NULL, summary_op = NULL) {
  if (!is.null(save_secs) && !is.null(save_steps)) {
    stop(" Only one of save_secs or save_steps can be specified")
  }
  tf$python$training$basic_session_run_hooks$SummarySaverHook(
    save_steps = as_nullable_integer(save_steps),
    save_secs = as_nullable_integer(save_secs),
    output_dir = output_dir,
    summary_writer = summary_writer,
    scaffold = scaffold,
    summary_op = summary_op
  )
}


#' Delay execution until global step reaches to wait_until_step.
#' 
#' This hook delays execution until global step reaches to `wait_until_step`. It
#' is used to gradually start workers in distributed settings. One example usage
#' would be setting `wait_until_step=int(K*log(task_id+1))` assuming that 
#' task_id=0 is the chief.
#' 
#' @param wait_until_step an `int` shows until which global step should we wait.
#'   
#' @family session_run_hook wrappers
#'   
#' @export
hook_global_step_waiter <- function(wait_until_step) {
  tf$python$training$basic_session_run_hooks$GlobalStepWaiterHook(
    wait_until_step = as.integer(wait_until_step)
  )
}

#' Base R6 class for creating custom SessionRunHook
#' 
#' @docType class
#' 
#' @format An [R6Class] generator object
#' 
#' @section Methods:
#' \describe{
#'  \item{\code{begin()}}{Called once before using the session.}
#'  \item{\code{after_create_session(session, coord)}}{Called when new TensorFlow session is created.}
#'  \item{\code{before_run(run_context)}}{Called before each call to run().}
#'  \item{\code{after_run(run_context, run_values)}}{Called after each call to run().}
#'  \item{\code{end(session)}}{Called at the end of session.}
#' }
#' 
#' @return [KerasCallback].
#' 
#' @examples 
#' library(tfestimators)
#' 
#' CustomSessionRunHook <- R6::R6Class(
#'   "CustomSessionRunHook",
#'   inherit = EstimatorSessionRunHook,
#'   public = list(
#'     end = function(session) {
#'       cat("Running custom session run hook at the end of a session")
#'     })
#'  )
#' custom_hook <- CustomSessionRunHook$new()
#'  
#' @family session_run_hook wrappers
#' @export
EstimatorSessionRunHook <- R6Class("EstimatorSessionRunHook",

                         public = list(

                           begin = function() {

                           },

                           after_create_session = function(session, coord) {

                           },

                           before_run = function(run_context) {

                           },

                           after_run = function(run_context, run_values) {

                           },

                           end = function(session) {

                           }
                         )
)

normalize_session_run_hooks <- function(session_run_hooks) {

  if (is.null(session_run_hooks))
    return(NULL)

  if (!is.null(session_run_hooks) && !is.list(session_run_hooks))
    session_run_hooks <- list(session_run_hooks)

  # import callback utility module
  python_path <- system.file("python", package = "tfestimators")
  tools <- import_from_path("estimatortools", path = python_path)

  lapply(session_run_hooks, function(session_run_hook) {
    if (inherits(session_run_hook, "EstimatorSessionRunHook")) {
      # create a python SessionRunHook to map to our R SessionRunHook
      tools$session_run_hooks$RSessionRunHook(
        r_begin = session_run_hook$begin,
        r_after_create_session = session_run_hook$after_create_session,
        r_before_run = session_run_hook$before_run,
        r_after_run = session_run_hook$after_run,
        r_end = session_run_hook$end
      )
    } else {
      session_run_hook
    }
  })
}
