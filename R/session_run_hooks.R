#' Prints Given Tensors Every N Local Steps, Every N Seconds, or at End
#' 
#' The tensors will be printed to the log, with `INFO` severity.
#' 
#' Note that if `at_end` is `TRUE`, `tensors` should not include any tensor 
#' whose evaluation produces a side effect such as consuming additional inputs.
#' 
#' @param tensors A list that maps string-valued tags to tensors/tensor names.
#' @param every_n_iter An integer value, indicating the values of `tensors` will be printed
#' once every N local steps taken on the current worker.
#' @param every_n_secs An integer or float value, indicating the values of `tensors` will be printed
#' once every N seconds. Exactly one of `every_n_iter` and `every_n_secs` should be provided.
#' @param formatter A function that takes `list(tag = tensor)` and returns a
#'   string. If `NULL` uses default printing all tensors.
#' @param at_end A boolean value specifying whether to print the values of `tensors` at the
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
  with_logging_verbosity(tf$logging$INFO, {
      tf$python$training$basic_session_run_hooks$LoggingTensorHook(
        tensors = ensure_dict(tensors, named = F),
        every_n_iter = every_n_iter,
        every_n_secs = every_n_secs,
        formatter = formatter,
        at_end = at_end
    )
  })
}

#' Monitor to Request Stop at a Specified Step
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


#' Saves Checkpoints Every N Steps or Seconds
#' 
#' @param checkpoint_dir The base directory for the checkpoint files.
#' @param save_secs An integer, indicating saving checkpoints every N secs.
#' @param save_steps An integer, indicating saving checkpoints every N steps.
#' @param saver A saver object, used for saving.
#' @param checkpoint_basename The base name for the checkpoint files.
#' @param scaffold A scaffold, used to get saver object.
#' @param listeners List of checkpoint saver listener subclass instances, used
#'   for callbacks that run immediately after the corresponding
#'   `hook_checkpoint_saver` callbacks, only in steps where `the hook_checkpoint_saver`
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


#' Steps per Second Monitor
#' 
#' @param every_n_steps Run this counter every N steps
#' @param every_n_secs Run this counter every N seconds
#' @param output_dir The output directory
#' @param summary_writer The summary writer
#' 
#' @family session_run_hook wrappers
#' 
#' @export
hook_step_counter <- function(every_n_steps = 100,
                              every_n_secs = NULL,
                              output_dir = NULL,
                              summary_writer = NULL) 
{
  tf$python$training$basic_session_run_hooks$StepCounterHook(
    every_n_steps = as.integer(every_n_steps),
    every_n_secs = as_nullable_integer(every_n_secs),
    output_dir = output_dir,
    summary_writer = summary_writer
  )
}

#' NaN Loss Monitor
#' 
#' Monitors loss and stops training if loss is NaN. Can either fail with
#' exception or just stop training.
#' 
#' @param loss_tensor The loss tensor.
#' @param fail_on_nan_loss A boolean indicating whether to raise exception when loss is NaN.
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

#' Saves Summaries Every N Steps
#' 
#' 
#' @param save_steps An integer indicating saving summaries every N steps. Exactly one of
#'   `save_secs` and `save_steps` should be set.
#' @param save_secs An integer indicating saving summaries every N seconds.
#' @param output_dir The directory to save the summaries to. Only used
#'   if no `summary_writer` is supplied.
#' @param summary_writer The summary writer. If `NULL` and an `output_dir` was
#'   passed, one will be created accordingly.
#' @param scaffold A scaffold to get summary_op if it's not provided.
#' @param summary_op A tensor of type `tf$string` containing the serialized
#'   summary protocol buffer or a list of tensors. They are most likely an
#'   output by TensorFlow summary methods like `tf$summary$scalar` or
#'   `tf$summary$merge_all`. It can be passed in as one tensor; if more than
#'   one, they must be passed in as a list.
#'   
#' @family session_run_hook wrappers
#'   
#' @export
hook_summary_saver <- function(save_steps = NULL,
                               save_secs = NULL,
                               output_dir = NULL,
                               summary_writer = NULL,
                               scaffold = NULL,
                               summary_op = NULL)
{
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


#' Delay Execution until Global Step Reaches to `wait_until_step`.
#' 
#' This hook delays execution until global step reaches to `wait_until_step`. It
#' is used to gradually start workers in distributed settings. One example usage
#' would be setting `wait_until_step=int(K*log(task_id+1))` assuming that 
#' `task_id=0` is the chief.
#' 
#' @param wait_until_step An integer indicating that until which global step should we wait.
#'   
#' @family session_run_hook wrappers
#'   
#' @export
hook_global_step_waiter <- function(wait_until_step) {
  tf$python$training$basic_session_run_hooks$GlobalStepWaiterHook(
    wait_until_step = as.integer(wait_until_step)
  )
}

#' TensorFlow Session Run Hook used in Estimators
#'
#' This is the base R6 class used for custom session run hooks, which can be
#' used to monitor estimators while they are trained by TensorFlow. This class
#' itself does not provide any hooks; users should define their own R6 classes
#' extending this class and override the methods as needed.
#' 
#' @docType class
#' 
#' @format An [R6Class] generator object.
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
EstimatorSessionRunHook <- R6Class(
  "EstimatorSessionRunHook",
  
  public = list(
    
    initialize = function(begin = NULL,
                          after_create_session = NULL,
                          before_run = NULL,
                          after_run = NULL,
                          end = NULL)
    {
      for (key in ls()) {
        object <- get(key, envir = environment())
        self[[key]] <- object %||% function(...) {}
      }
    }
  ),
  
  lock_objects = FALSE
)

#' Create Session Run Hooks
#' 
#' Create a set of session run hooks, used to record information during
#' training of an estimator.
#' 
#' @param begin `function()`: An \R function, to be called once before using the session.
#' @param after_create_session `function(session, coord)`: An \R function, to be called
#'   once the new TensorFlow session has been created.
#' @param before_run `function(run_context)`: An \R function to be called before a run.
#' @param after_run `function(run_context, run_values)`: An \R function to be called
#'   after a run.
#' @param end `function(session)`: An \R function to be called at the end of the session.
#' 
#' @export
session_run_hook <- function(begin = NULL,
                             after_create_session = NULL,
                             before_run = NULL,
                             after_run = NULL,
                             end = NULL)
{
  EstimatorSessionRunHook$new(
    begin = begin,
    after_create_session = after_create_session,
    before_run = before_run,
    after_run = after_run,
    end = end
  )
}

normalize_session_run_hooks <- function(session_run_hooks) {

  if (is.null(session_run_hooks))
    return(NULL)

  session_run_hooks <- ensure_nullable_list(session_run_hooks)

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
