tf_custom_estimator <- function(estimator, model_fn, classes) {
  structure(
    list(
      estimator = estimator,
      model_fn  = model_fn
    ),
    class = c("tf_estimator", "tf_custom_estimator", classes)
  )
}

#' Define an Estimator Specification
#' 
#' Define the estimator specification, used as part of the `model_fn` defined with
#' custom estimators created by [estimator()]. See [estimator()] for more details.
#' 
#' @param mode A key that specifies whether we are performing
#'   training (`"train"`), evaluation (`"eval"`), or prediction (`"infer"`).
#'   These values can also be accessed through the [mode_keys()] object.
#'   
#' @param predictions The prediction tensor(s).
#' 
#' @param loss The training loss tensor. Must be either scalar, or with shape `c(1)`.
#' 
#' @param train_op The training operation -- typically, a call to `optimizer$minimize(...)`,
#'   depending on the type of optimizer used during training.
#'   
#' @param eval_metric_ops A list of metrics to be computed as part of evaluation.
#'   This should be a named list, mapping metric names (e.g. `"rmse"`) to the operation
#'   that computes the associated metric (e.g. `tf$metrics$root_mean_squared_error(...)`).
#'   These metric operations should be evaluated without any impact on state (typically 
#'   is a pure computation results based on variables). For example, it should not
#'   trigger the update ops or requires any input fetching.
#'
#' @param training_hooks (Available since TensorFlow v1.4) A list of session run hooks to run on all workers during training.
#' 
#' @param evaluation_hooks (Available since TensorFlow v1.4) A list of session run hooks to run during evaluation.
#' 
#' @param training_chief_hooks (Available since TensorFlow v1.4) A list of session run hooks to run on chief worker during training.
#' 
#' @param ... Other optional (named) arguments, to be passed to the `EstimatorSpec` constructor.
#' 
#' @export
#' @family custom estimator methods
estimator_spec <- function(mode,
                           predictions = NULL,
                           loss = NULL,
                           train_op = NULL,
                           eval_metric_ops = NULL,
                           training_hooks = NULL,
                           evaluation_hooks = NULL,
                           training_chief_hooks = NULL,
                           ...)
{
  args <- list(
    mode = mode,
    predictions = predictions,
    loss = loss,
    train_op = train_op,
    # TODO: need to use reticulate::tuple() - fix this on Python end to soften the requirements in model_fn
    eval_metric_ops = reticulate::dict(
      lapply(eval_metric_ops, function(x) reticulate::tuple(unlist(x)))),
    ...
  )
  if (tf_version() >= '1.4') {
    args$training_hooks <- training_hooks
    args$evaluation_hooks <- evaluation_hooks
    args$training_chief_hooks <- training_chief_hooks
  }
  do.call(estimator_lib$model_fn_lib$EstimatorSpec, args)
}

#' Construct a Custom Estimator
#' 
#' Construct a custom estimator, to be used to train and evaluate
#' TensorFlow models.
#' 
#' The `Estimator` object wraps a model which is specified by a `model_fn`, 
#' which, given inputs and a number of other parameters, returns the operations
#' necessary to perform training, evaluation, and prediction.
#' 
#' All outputs (checkpoints, event files, etc.) are written to `model_dir`, or a
#' subdirectory thereof. If `model_dir` is not set, a temporary directory is 
#' used.
#' 
#' The `config` argument can be used to passed run configuration object
#' containing information about the execution environment. It is passed on to
#' the `model_fn`, if the `model_fn` has a parameter named "config" (and input
#' functions in the same manner). If the `config` parameter is not passed, it is
#' instantiated by `estimator()`. Not passing config means that defaults useful
#' for local execution are used. `estimator()` makes config available to the
#' model (for instance, to allow specialization based on the number of workers
#' available), and also uses some of its fields to control internals, especially
#' regarding checkpointing.
#' 
#' The `params` argument contains hyperparameters. It is passed to the 
#' `model_fn`, if the `model_fn` has a parameter named "params", and to the
#' input functions in the same manner. `estimator()` only passes `params` along, it
#' does not inspect it. The structure of `params` is therefore entirely up to
#' the developer.
#' 
#' None of estimator's methods can be overridden in subclasses (its 
#' constructor enforces this). Subclasses should use `model_fn` to configure the
#' base class, and may add methods implementing specialized functionality.
#' 
#' @section Model Functions:
#' 
#' The `model_fn` should be an \R function of the form:
#' \preformatted{function(features, labels, mode, params) {
#'     # 1. Configure the model via TensorFlow operations.
#'     # 2. Define the loss function for training and evaluation.
#'     # 3. Define the training optimizer.
#'     # 4. Define how predictions should be produced.
#'     # 5. Return the result as an `estimator_spec()` object.
#'     estimator_spec(mode, predictions, loss, train_op, eval_metric_ops)
#' }}
#' 
#' The model function's inputs are defined as follows:
#' 
#' \tabular{ll}{
#' `features` \tab
#' The feature tensor(s). \cr
#' `labels`   \tab
#' The label tensor(s). \cr
#' `mode`     \tab
#' The current training mode ("train", "eval", "infer").
#' These can be accessed through the `mode_keys()` object. \cr
#' `params`   \tab
#' An optional list of hyperparameters, as received
#' through the `estimator()` constructor. \cr
#' }
#' 
#' See [estimator_spec()] for more details as to how the estimator specification
#' should be constructed, and <https://www.tensorflow.org/extend/estimators#constructing_the_model_fn> for
#' more information as to how the model function should be constructed.
#' 
#' @param model_fn The model function. See **Model Function** for details
#'   on the structure of a model function.
#' @param model_dir Directory to save model parameters, graph and etc. This can
#'   also be used to load checkpoints from the directory into a estimator to
#'   continue training a previously saved model. If `NULL`, the `model_dir` in
#'   `config` will be used if set. If both are set, they must be same. If both
#'   are `NULL`, a temporary directory will be used.
#' @param config Configuration object.
#' @param params List of hyper parameters that will be passed into `model_fn`.
#'   Keys are names of parameters, values are basic python types.
#' @param class An optional set of \R classes to add to the generated object.
#'
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


