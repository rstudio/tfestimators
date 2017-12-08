#' Train and evaluate the estimator.
#' 
#' (Available since TensorFlow v1.4)
#' 
#' This utility function trains, evaluates, and (optionally) exports the model by
#' using the given `estimator`. All training related specification is held in
#' `train_spec`, including training `input_fn` and training max steps, etc. All
#' evaluation and export related specification is held in `eval_spec`, including
#' evaluation `input_fn`, steps, etc.
#' 
#' This utility function provides consistent behavior for both local
#' (non-distributed) and distributed configurations. Currently, the only
#' supported distributed training configuration is between-graph replication.
#' 
#' Overfitting: In order to avoid overfitting, it is recommended to set up the
#' training `input_fn` to shuffle the training data properly. It is also
#' recommended to train the model a little longer, say multiple epochs, before
#' performing evaluation, as the input pipeline starts from scratch for each
#' training. It is particularly important for local training and evaluation.
#' 
#' Stop condition: In order to support both distributed and non-distributed
#' configuration reliably, the only supported stop condition for model
#' training is `train_spec.max_steps`. If `train_spec.max_steps` is `NULL`, the
#' model is trained forever. *Use with care* if model stop condition is
#' different. For example, assume that the model is expected to be trained with
#' one epoch of training data, and the training `input_fn` is configured to throw
#' `OutOfRangeError` after going through one epoch, which stops the
#' `Estimator.train`. For a three-training-worker distributed configuration, each
#' training worker is likely to go through the whole epoch independently. So, the
#' model will be trained with three epochs of training data instead of one epoch.
#' 
#' 
#' @param object An estimator object to train and evaluate.
#' @param train_spec A `TrainSpec` instance to specify the training specification.
#' @param eval_spec A `EvalSpec` instance to specify the evaluation and export specification.
#' 
#' @section Raises:
#' * ValueError: if environment variable `TF_CONFIG` is incorrectly set.
#' 
#' @family training methods
#' @export
train_and_evaluate.tf_estimator <- function(object, train_spec, eval_spec) {
  if (tf_version() < '1.4') {
    stop("train_and_evaluate() is only available since TensorFlow v1.4")
  }
  estimator <- object$estimator
  train_spec$args$input_fn <- normalize_input_fn(object, train_spec$args$input_fn)
  eval_spec$args$input_fn <- normalize_input_fn(object, eval_spec$args$input_fn)
  with_logging_verbosity(tf$logging$WARN, {
    tf$estimator$train_and_evaluate(
      estimator = estimator,
      train_spec = do.call(tf$estimator$TrainSpec, train_spec$args),
      eval_spec = do.call(tf$estimator$EvalSpec, eval_spec$args)
    )
  })
  
  training_history <- as.data.frame(.globals$history[[mode_keys()$TRAIN]])
  steps_per_axis_unit <- training_history$steps[2] - training_history$steps[1]
  
  training_history <- if (nrow(training_history) > 100) {
    # cap number of points plotted
    nrow_history <- nrow(training_history)
    sampling_indices <- seq(1, nrow_history, by = nrow_history / 100) %>%
      as.integer()
    num_steps <- training_history$steps %>%
      tail(1)
    steps_per_axis_unit <<- num_steps / 100
    training_history[sampling_indices, names(training_history) != "steps"]
  } else training_history[, names(training_history) != "steps"]
  tfruns::write_run_metadata("metrics", training_history)
  
  properties <- list()
  properties$steps_per_axis_unit <- steps_per_axis_unit
  tfruns::write_run_metadata("properties", properties)
  
  evaluation_results <- as.data.frame(.globals$history[[mode_keys()$EVAL]]) %>%
    tail(1) %>%
    as.list()
  tfruns::write_run_metadata("evaluation", evaluation_results)
  invisible(training_history)
}


#' Configuration for the train component of `train_and_evaluate`
#' 
#' `TrainSpec` determines the input data for the training, as well as the
#' duration. Optional hooks run at various stages of training.
#' 
#' @param input_fn Training input function returning a tuple of:
#' * features - `Tensor` or dictionary of string feature name to `Tensor`.
#' * labels - `Tensor` or dictionary of `Tensor` with labels.
#' @param max_steps Positive number of total steps for which to train model.
#' If `NULL`, train forever. The training `input_fn` is not expected to
#' generate `OutOfRangeError` or `StopIteration` exceptions.
#' @param hooks List of session run hooks to run on all workers
#' (including chief) during training.
#' 
#' @family training methods
#' @export
train_spec <- function(input_fn,
                       max_steps = NULL,
                       hooks = NULL) {
  structure(
    list(
      args = list(
        input_fn = input_fn,
        max_steps = as_nullable_integer(max_steps),
        hooks = resolve_train_hooks(hooks, max_steps)
      )
    ),
    class = "train_spec"
  )
}

#' Configuration for the eval component of `train_and_evaluate`
#' 
#' `EvalSpec` combines details of evaluation of the trained model as well as its
#' export. Evaluation consists of computing metrics to judge the performance of
#' the trained model. Export writes out the trained model on to external
#' storage.
#' 
#' @param input_fn Evaluation input function returning a tuple of:
#' * features - `Tensor` or dictionary of string feature name to `Tensor`.
#' * labels - `Tensor` or dictionary of `Tensor` with labels.
#' @param steps Positive number of steps for which to evaluate model.
#' If `NULL`, evaluates until `input_fn` raises an end-of-input exception.
#' @param name Name of the evaluation if user needs to run multiple
#' evaluations on different data sets. Metrics for different evaluations
#' are saved in separate folders, and appear separately in tensorboard.
#' @param hooks List of session run hooks to run
#' during evaluation.
#' @param exporters List of `Exporter`s, or a single one, or `NULL`.
#' `exporters` will be invoked after each evaluation.
#' @param start_delay_secs Start evaluating after waiting for this many
#' seconds.
#' @param throttle_secs Do not re-evaluate unless the last evaluation was
#' started at least this many seconds ago. Of course, evaluation does not
#' occur if no new checkpoints are available, hence, this is the minimum.
#' 
#' @family training methods
#' @export
eval_spec <- function(input_fn,
                      steps = 100,
                      name = NULL,
                      hooks = NULL,
                      exporters = NULL,
                      start_delay_secs = 120,
                      throttle_secs = 600) {
  structure(
    list(
      args = list(
        input_fn = input_fn,
        steps = as_nullable_integer(steps),
        name = name,
        hooks = resolve_eval_hooks(hooks, steps),
        exporters = exporters,
        start_delay_secs = as.integer(start_delay_secs),
        throttle_secs = as.integer(throttle_secs)
      )
    ),
    class = "eval_spec"
  )
}


