#' @export
tf_experiment <- function(name, ...) {
  object <- list(...)
  class(object) <- "tf_experiment"
  object
}

#' @export
experiment <- function(x, ...) {
  UseMethod("experiment")
}

#' @export
experiment.tf_custom_model <- function(object, ...) {
  experiment.tf_model(object, ...)
}

#' Interleaves training and evaluation.
#' 
#' The frequency of evaluation is controlled by the contructor arg
#' `min_eval_frequency`. When this parameter is 0, evaluation happens
#' only after training has completed. Note that evaluation cannot happen
#' more frequently than checkpoints are taken. If no new snapshots are
#' available when evaluation is supposed to occur, then evaluation doesn't
#' happen for another `min_eval_frequency` steps (assuming a checkpoint is
#' available at that point). Thus, settings `min_eval_frequency` to 1 means
#' that the model will be evaluated everytime there is a new checkpoint. This is particular useful for a "Master" task in the cloud, whose
#' responsibility it is to take checkpoints, evaluate those checkpoints,
#' and write out summaries. Participating in training as the supervisor
#' allows such a task to accomplish the first and last items, while
#' performing evaluation allows for the second. Returns: The result of the `evaluate` call to the `Estimator` as well as the export results using the specified `ExportStrategy`.
#' 
#' 
#' @return The result of the `evaluate` call to the `Estimator` as well as the export results using the specified `ExportStrategy`.
#' 
#' @export
#' @family experiment methods
train_and_evaluate.tf_experiment <- function(object) {
  object$experiment$train_and_evaluate()
}

#' Evaluate on the evaluation data.
#' 
#' Runs evaluation on the evaluation data and returns the result. Runs for
#' `self._eval_steps` steps, or if it's `NULL`, then run until input is
#' exhausted or another exception is raised. Start the evaluation after
#' `delay_secs` seconds, or if it's `NULL`, defaults to using
#' `self._eval_delay_secs` seconds.
#' 
#' @param delay_secs Start evaluating after this many seconds. If `NULL`, defaults to using `self._eval_delays_secs`.
#' 
#' @return The result of the `evaluate` call to the `Estimator`.
#' 
#' @export
#' @family experiment methods
evaluate.tf_experiment <- function(object, delay_secs = NULL) {
  object$experiment$evaluate(delay_secs = delay_secs)
}


#' Fit the estimator using the training data.
#' 
#' Train the estimator for `self._train_steps` steps, after waiting for
#' `delay_secs` seconds. If `self._train_steps` is `NULL`, train forever.
#' 
#' @param delay_secs Start training after this many seconds.
#' 
#' @return The trained estimator.
#' @export
#' @family experiment methods
train.tf_experiment <- function(object, delay_secs = NULL) {
  object$experiment$train(delay_secs = delay_secs)
}

# TODO: Need to generate doc and args -  reticulate::py_function_wrapper("tf$contrib$learn$Experiment")
# Doc not generated correctly
#' @export
experiment.tf_model <- function(object,
                                train_input_fn,
                                eval_input_fn,
                                train_steps = 2L,
                                eval_steps = 2L,
                                ...) {
  
  exp <- tf$contrib$learn$Experiment(
    estimator = object$estimator,
    train_input_fn = train_input_fn$input_fn,
    eval_input_fn = eval_input_fn$input_fn,
    train_steps = train_steps,
    eval_steps = eval_steps,
    ...)
  tf_experiment(experiment = exp)
}
