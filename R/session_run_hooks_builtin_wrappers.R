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
        tensors = ensure_dict(tensors, named = FALSE),
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
    num_steps = ensure_scalar_integer(num_steps, allow.null = TRUE),
    last_step = ensure_scalar_integer(last_step, allow.null = TRUE)
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
    save_secs = ensure_scalar_integer(save_secs, allow.null = TRUE),
    save_steps = ensure_scalar_integer(save_steps, allow.null = TRUE),
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
    every_n_steps = ensure_scalar_integer(every_n_steps, allow.null = TRUE),
    every_n_secs = ensure_scalar_integer(every_n_secs, allow.null = TRUE),
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
    save_steps = ensure_scalar_integer(save_steps, allow.null = TRUE),
    save_secs = ensure_scalar_integer(save_secs, allow.null = TRUE),
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
    wait_until_step = ensure_scalar_integer(wait_until_step)
  )
}

#' Create Session Run Arguments
#' 
#' Create a set of session run arguments. These are used as the return values in
#' the `before_run(context)` callback of a [session_run_hook()], for requesting
#' the values of specific tensor in the `after_run(context, values)` callback.
#' 
#' @param ... A set of tensors or operations.
#' 
#' @seealso [session_run_hook()]
#' @export
session_run_args <- function(...) {
  tf$train$SessionRunArgs(list(...))
}

#' Create Custom Session Run Hooks
#' 
#' Create a set of session run hooks, used to record information during
#' training of an estimator. See **Details** for more information on the
#' various hooks that can be defined.
#' 
#' @param begin `function()`: An \R function, to be called once before using the session.
#' @param after_create_session `function(session, coord)`: An \R function, to be called
#'   once the new TensorFlow session has been created.
#' @param before_run `function(run_context)`: An \R function to be called before a run.
#' @param after_run `function(run_context, run_values)`: An \R function to be called
#'   after a run.
#' @param end `function(session)`: An \R function to be called at the end of the session.
#' 
#' Typically, you'll want to define a `before_run()` hook that defines the set
#' of tensors you're interested in for a particular run, and then you'll use the
#' resulting values of those tensors in your `after_run()` hook. The tensors
#' requested in your `before_run()` hook will be made available as part of the
#' second argument in the `after_run()` hook (the `values` argument).
#' 
#' @seealso [session_run_args()]
#' @family session_run_hook wrappers
#' @export
session_run_hook <- function(
  begin = function() {},
  after_create_session = function(session, coord) {},
  before_run = function(context) {},
  after_run = function(context, values) {},
  end = function(session) {})
{
  envir <- new.env(parent = emptyenv())
  
  return_null <- function(callback, ...) {
    force(callback)
    function(...) {
      callback(...)
      NULL
    }
  }
  
  envir$begin                <- return_null(begin)
  envir$after_create_session <- return_null(after_create_session)
  envir$before_run           <- before_run
  envir$after_run            <- return_null(after_run)
  envir$end                  <- return_null(end)
  
  class(envir) <- "EstimatorSessionRunHook"
  envir
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
