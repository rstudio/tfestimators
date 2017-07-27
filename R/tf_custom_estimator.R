tf_custom_estimator <- function(estimator, model_fn, classes) {
  structure(
    list(
      estimator = estimator,
      model_fn  = model_fn
    ),
    class = c("tf_estimator", "tf_custom_estimator", classes)
  )
}

#' Ops and objects returned from a `model_fn` and passed to `Estimator`.
#' 
#' `EstimatorSpec` fully defines the model to be run by `Estimator`.
#' 
#' @param mode A `ModeKeys`. Specifies if this is training, evaluation or prediction.
#' @param predictions Predictions `Tensor` or dict of `Tensor`.
#' @param loss Training loss `Tensor`. Must be either scalar, or with shape `c(1)`.
#' @param train_op Op for the training step.
#' @param eval_metric_ops Dict of metric results keyed by name. The values of the dict are the results of calling a metric function,
#' namely a `(metric_tensor, update_op)` list.
#' @param export_outputs Describes the output signatures to be exported to `SavedModel` and used during serving. 
#' A dict `{name: output}` where:
#' * name: An arbitrary name for this output. 
#' * output: an `ExportOutput` object such as `ClassificationOutput`, `RegressionOutput`, or `PredictOutput`. 
#' Single-headed models only need to specify one entry in this dictionary. 
#' Multi-headed models should specify one entry for each head, one of which must be named using
#' signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY.
#' @param training_chief_hooks Iterable of `tf.train.SessionRunHook` objects to run on the chief worker during training.
#' @param training_hooks Iterable of `tf.train.SessionRunHook` objects that to run on all workers during training.
#' @param scaffold A `tf.train.Scaffold` object that can be used to set initialization, saver, and more to be used in training.
#' 
#' @export
#' @family custom estimator methods
estimator_spec <- function(mode,
                           predictions = NULL,
                           loss = NULL,
                           train_op = NULL,
                           eval_metric_ops = NULL,
                           export_outputs = NULL,
                           training_chief_hooks = NULL,
                           training_hooks = NULL,
                           scaffold = NULL)
{
  estimator_lib$model_fn_lib$EstimatorSpec(
    mode = mode,
    predictions = predictions,
    loss = loss,
    train_op = train_op,
    # TODO: need to use reticulate::tuple() - fix this on Python end to soften the requirements in model_fn
    eval_metric_ops = reticulate::dict(
      lapply(eval_metric_ops, function(x) reticulate::tuple(unlist(x)))),
    export_outputs = export_outputs,
    training_chief_hooks = training_chief_hooks,
    training_hooks = training_hooks,
    scaffold = scaffold)
}

#' Custom estimator constructor
#' 
#' This is the core Estimator class to train and evaluate TensorFlow models.
#' 
#' The `Estimator` object wraps a model which is specified by a `model_fn`, 
#' which, given inputs and a number of other parameters, returns the ops 
#' necessary to perform training, evaluation, or predictions.
#' 
#' All outputs (checkpoints, event files, etc.) are written to `model_dir`, or a
#' subdirectory thereof. If `model_dir` is not set, a temporary directory is 
#' used.
#' 
#' The `config` argument can be passed `RunConfig` object containing information
#' about the execution environment. It is passed on to the `model_fn`, if the 
#' `model_fn` has a parameter named "config" (and input functions in the same 
#' manner). If the `config` parameter is not passed, it is instantiated by the 
#' `Estimator`. Not passing config means that defaults useful for local
#' execution are used. `Estimator` makes config available to the model (for
#' instance, to allow specialization based on the number of workers available),
#' and also uses some of its fields to control internals, especially regarding
#' checkpointing.
#' 
#' The `params` argument contains hyperparameters. It is passed to the 
#' `model_fn`, if the `model_fn` has a parameter named "params", and to the
#' input functions in the same manner. `Estimator` only passes params along, it
#' does not inspect it. The structure of `params` is therefore entirely up to
#' the developer.
#' 
#' None of `Estimator`'s methods can be overridden in subclasses (its 
#' constructor enforces this). Subclasses should use `model_fn` to configure the
#' base class, and may add methods implementing specialized functionality.
#' 
#' @param model_fn Model function. Follows the signature: 
#'   * `features`: single `Tensor` or `dict` of `Tensor`s (depending on data 
#'     passed to `train`), 
#'   * `labels`: `Tensor` or `dict` of `Tensor`s (for multi-head models). 
#'     If mode is `ModeKeys.PREDICT`, `labels=NULL` will be passed. 
#'     If the `model_fn`'s signature does not accept `mode`, the `model_fn` 
#'     must still be able to handle `labels=NULL`. 
#'   * `mode`: Optional. Specifies if this training, evaluation or prediction. 
#'     See `ModeKeys`. 
#'   * `params`: Optional `dict` of hyperparameters. Will receive what is passed 
#'     to Estimator in `params` parameter. This allows to configure Estimators 
#'     from hyper parameter tuning. 
#'   * `config`: Optional configuration object. Will receive what is passed to 
#'   Estimator in `config` parameter, or the default `config`. Allows updating 
#'   things in your model_fn based on configuration such as `num_ps_replicas`, or `model_dir`.
#' @param model_dir Directory to save model parameters, graph and etc. This can
#'   also be used to load checkpoints from the directory into a estimator to
#'   continue training a previously saved model. If `NULL`, the model_dir in
#'   `config` will be used if set. If both are set, they must be same. If both
#'   are `NULL`, a temporary directory will be used.
#' @param config Configuration object.
#' @param params `dict` of hyper parameters that will be passed into `model_fn`.
#'   Keys are names of parameters, values are basic python types.
#' @param class An optional set of \R classes to add to the generated object.
#'   
#' @export
#' @family custom estimator methods
estimator <- function(model_fn,
                      model_dir = NULL,
                      config = NULL,
                      params = NULL,
                      class = NULL)
{
  model_fn <- as_model_fn(model_fn)
  estimator <- py_suppress_warnings(estimator_lib$Estimator(
    model_fn = model_fn,
    model_dir = resolve_model_dir(model_dir),
    config = config,
    params = params
  ))
  tf_custom_estimator(estimator, model_fn, class)
}

as_model_fn <- function(f) {
  tools <- import_package_module("estimatortools.functions")
  tools$as_model_fn(f)
}


with_logging_verbosity <- function(verbosity, expr) {
  old <- tf$logging$get_verbosity()
  on.exit(tf$logging$set_verbosity(old), add = TRUE)
  tf$logging$set_verbosity(verbosity)
  force(expr)
}


