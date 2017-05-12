#' @export
tf_experiment <- function(name, ...) {
  object <- list(...)
  class(object) <- "tf_experiment"
  object
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

#' Experiment constructor
#' 
#' An Experiment contains all information needed to train a model.
#' After an experiment is created (by passing an Estimator and inputs for
#' training and evaluation), an Experiment instance knows how to invoke training
#' and eval loops in a sensible fashion for distributed training.
#' @param train_input_fn function, returns features and labels for training.
#' @param eval_input_fn function, returns features and labels for evaluation. 
#' If`eval_steps` is `None`, this should be configured only to produce for a
#' finite number of batches (generally, 1 epoch over the evaluation data).
#' @param eval_metrics `list` of string, metric function. If `NULL`, default
#' set is used. This should be `NULL` if the `estimator` is `tf.estimator.Estimator`.
#' If metrics are provided they will be *appended* to the default set.
#' @param train_steps Perform this many steps of training. `NULL`, the default, means train forever.
#' @param eval_steps `evaluate` runs until input is exhausted (or another exception is raised),
#' or for `eval_steps` steps, if specified.
#' @param train_monitors A list of monitors to pass to the `Estimator`'s `fit` function.
#' @param eval_hooks A list of `SessionRunHook` hooks to pass to the `Estimator`'s
#' `evaluate` function.
#' @param local_eval_frequency (applies only to local_run) Frequency of running 
#' eval in steps. If `None`, runs evaluation only at the end of training.
#' @param eval_delay_secs Start evaluating after waiting for this many seconds.
#' @param continuous_eval_throttle_secs Do not re-evaluate unless the last evaluation
#' was started at least this many seconds ago for continuous_eval().
#' @param min_eval_frequency: (applies only to train_and_evaluate). the minimum
#' number of steps between evaluations. Of course, evaluation does not
#' occur if no new snapshot is available, hence, this is the minimum.
#' If 0, the evaluation will only happen after training.
#' If NULL, defaults to 1, unless model_dir is on GCS, in which case the default is 1000.
#' @param delay_workers_by_global_step if `TRUE` delays training workers based 
#' on global step instead of time.
#' @param export_strategies Iterable of `ExportStrategy`s, or a single one, or `NULL`.
#' @param train_steps_per_iteration (applies only to continuous_train_and_eval). 
#' Perform this many (integer) number of train steps for each training-evaluation
#' iteration. With a small value, the model will be evaluated more frequently 
#' with more checkpoints saved. If `NULL`, will use a default value (which is
#' smaller than `train_steps` if provided).
#' @export
#' @family experiment methods
experiment.tf_model <- function(object,
                                train_input_fn,
                                eval_input_fn,
                                train_steps = 2L,
                                eval_steps = 2L,
                                eval_metrics = NULL,
                                train_monitors = NULL,
                                eval_hooks = NULL,
                                local_eval_frequency = NULL,
                                eval_delay_secs = 120L,
                                continuous_eval_throttle_secs = 60L,
                                min_eval_frequency = NULL,
                                delay_workers_by_global_step = NULL,
                                export_strategies = NULL,
                                train_steps_per_iteration = NULL) {
  exp <- contrib_learn$Experiment(
    estimator = object$estimator,
    train_input_fn = train_input_fn(is.tf_model(object)),
    eval_input_fn = eval_input_fn(is.tf_model(object)),
    train_steps = as.integer(train_steps),
    eval_steps = as.integer(eval_steps),
    eval_metrics = eval_metrics,
    train_monitors = train_monitors,
    eval_hooks = eval_hooks,
    local_eval_frequency = local_eval_frequency,
    eval_delay_secs = as.integer(eval_delay_secs),
    continuous_eval_throttle_secs = as.integer(continuous_eval_throttle_secs),
    min_eval_frequency = as_nullable_integer(min_eval_frequency),
    delay_workers_by_global_step = as_nullable_integer(delay_workers_by_global_step),
    export_strategies = export_strategies,
    train_steps_per_iteration = as_nullable_integer(train_steps_per_iteration))
  tf_experiment(experiment = exp)
}
