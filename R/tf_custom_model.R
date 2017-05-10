tf_custom_model <- function(...) {
  object <- list(...)
  class(object) <- "tf_custom_model"
  object
}

validate_custom_model_input_fn <- function(input_fn) {
  validate_input_fn(input_fn)
  if (input_fn$features_as_named_list) {
    stop("The argument features_as_named_list in your input_fn must be FALSE for custom model")
  }
}

is.tf_custom_model <- function(object) {
  inherits(object, "tf_custom_model")
}

#' Ops and objects returned from a `model_fn` and passed to `Estimator`.
#' 
#' `EstimatorSpec` fully defines the model to be run by `Estimator`.
#' 
#' 
#' @export
#' @family custom estimator methods
estimator_spec <- function(predictions,
                           loss,
                           train_op,
                           mode) {
  estimator_lib$model_fn_lib$EstimatorSpec(
    mode = mode,
    predictions = predictions,
    loss = loss,
    train_op = train_op)
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
#' `Estimator`. Not passing config means that defaults useful for local execution
#' are used. `Estimator` makes config available to the model (for instance, to
#' allow specialization based on the number of workers available), and also uses
#' some of its fields to control internals, especially regarding checkpointing. 
#' 
#' The `params` argument contains hyperparameters. It is passed to the
#' `model_fn`, if the `model_fn` has a parameter named "params", and to the input
#' functions in the same manner. `Estimator` only passes params along, it does
#' not inspect it. The structure of `params` is therefore entirely up to the
#' developer. 
#' 
#' None of `Estimator`'s methods can be overridden in subclasses (its
#' constructor enforces this). Subclasses should use `model_fn` to configure
#' the base class, and may add methods implementing specialized functionality.
#' 
#' @param model_fn Model function. Follows the signature: 
#' * Args: 
#' * `features`: single `Tensor` or `dict` of `Tensor`s (depending on data passed to `train`), 
#' * `labels`: `Tensor` or `dict` of `Tensor`s (for multi-head models). If mode is `ModeKeys.PREDICT`, `labels=NULL` will be passed. 
#' If the `model_fn`'s signature does not accept `mode`, the `model_fn` must still be able to handle `labels=NULL`. 
#' * `mode`: Optional. Specifies if this training, evaluation or prediction. See `ModeKeys`. 
#' * `params`: Optional `dict` of hyperparameters. Will receive what is passed to Estimator in `params` parameter. 
#' This allows to configure Estimators from hyper parameter tuning. 
#' * `config`: Optional configuration object. Will receive what is passed to Estimator in `config` parameter, or the default `config`. 
#' Allows updating things in your model_fn based on configuration such as `num_ps_replicas`, or `model_dir`.
#' @param model_dir Directory to save model parameters, graph and etc. This can also be used to load checkpoints from the directory 
#' into a estimator to continue training a previously saved model. 
#' If `NULL`, the model_dir in `config` will be used if set. If both are set, they must be same.
#' If both are `NULL`, a temporary directory will be used.
#' @param config Configuration object.
#' @param params `dict` of hyper parameters that will be passed into `model_fn`. Keys are names of parameters, values are basic python types.
#' 
#' @export
#' @family custom estimator methods
estimator <- function(model_fn,
                      model_dir = NULL,
                      config = NULL,
                      params = NULL)
{
  model_fn <- as_model_fn(model_fn)
  est <- estimator_lib$Estimator(
    model_fn = model_fn,
    model_dir = model_dir,
    config = config,
    params = params
  )
  tf_custom_model(estimator = est, model_fn = model_fn)
}

#' Trains a model given training data input_fn.
#' 
#' 
#' 
#' @param input_fn Input function returning a list of: features - `Tensor` or dictionary of string feature name to `Tensor`. labels - `Tensor` or dictionary of `Tensor` with labels.
#' @param hooks List of `SessionRunHook` subclass instances. Used for callbacks inside the training loop.
#' @param steps Number of steps for which to train model. If `NULL`, train forever or train until input_fn generates the `OutOfRange` or `StopIteration` error. 'steps' works incrementally. If you call two times train(steps=10) then training occurs in total 20 steps. If `OutOfRange` or `StopIteration` error occurs in the middle, training stops before 20 steps. If you don't want to have incremental behaviour please set `max_steps` instead. If set, `max_steps` must be `NULL`.
#' @param max_steps Number of total steps for which to train model. If `NULL`, train forever or train until input_fn generates the `OutOfRange` or `StopIteration` error. If set, `steps` must be `NULL`. If `OutOfRange` or `StopIteration` error occurs in the middle, training stops before `max_steps` steps.
#' 
#' @return `self`, for chaining.
#' 
#' @section Raises:
#' ValueError: If both `steps` and `max_steps` are not `NULL`. ValueError: If either `steps` or `max_steps` is <= 0.
#' 
#' @export
#' @family custom estimator methods
fit.tf_custom_model <- function(object, input_fn, steps = NULL, hooks = NULL, max_steps = NULL) {
  validate_custom_model_input_fn(input_fn)
  object$estimator$train(
    input_fn = input_fn$input_fn,
    steps = as_nullable_integer(steps),
    hooks = hooks,
    max_steps = as_nullable_integer(max_steps))
  object
}

#' Returns predictions for given features.
#' 
#' @param input_fn Input function returning features which is a dictionary of string feature name to `Tensor` or `SparseTensor`. If it returns a list, first item is extracted as features. Prediction continues until `input_fn` raises an end-of-input exception (`OutOfRangeError` or `StopIteration`).
#' @param predict_keys list of `str`, name of the keys to predict. It is used if the `EstimatorSpec.predictions` is a `dict`. If `predict_keys` is used then rest of the predictions will be filtered from the dictionary. If `NULL`, returns all.
#' @param hooks List of `SessionRunHook` subclass instances. Used for callbacks inside the prediction call.
#' @param checkpoint_path Path of a specific checkpoint to predict. If `NULL`, the latest checkpoint in `model_dir` is used.
#' 
#' @section Yields:
#' Evaluated values of `predictions` tensors.
#' 
#' @section Raises:
#' ValueError: Could not find a trained model in model_dir. ValueError: if batch length of predictions are not same. ValueError: If there is a conflict between `predict_keys` and `predictions`. For example if `predict_keys` is not `NULL` but `EstimatorSpec.predictions` is not a `dict`.
#' 
#' @export
#' @family custom estimator methods
predict.tf_custom_model <- function(object,
                                    input_fn,
                                    checkpoint_path = NULL,
                                    predict_keys = NULL,
                                    hooks = NULL,
                                    as_iterable = F) {
  validate_custom_model_input_fn(input_fn)
  est <- object$estimator
  predictions <- est$predict(
    input_fn = input_fn$input_fn,
    checkpoint_path = checkpoint_path,
    hooks = hooks,
    predict_keys = predict_keys)
  if (!as_iterable) {
    if (!any(inherits(predictions, "python.builtin.iterator"),
             inherits(predictions, "python.builtin.generator"))) {
      warning("predictions are not iterable, no need to convert again")
    } else {
      predictions <- predictions %>% iterate
    }
  }
  predictions
}

#' Evaluates the model given evaluation data input_fn.
#' 
#' For each step, calls `input_fn`, which returns one batch of data.
#' Evaluates until:
#' - `steps` batches are processed, or
#' - `input_fn` raises an end-of-input exception (`OutOfRangeError` or
#' `StopIteration`).
#' 
#' @param input_fn Input function returning a list of: features - Dictionary of string feature name to `Tensor` or `SparseTensor`. labels - `Tensor` or dictionary of `Tensor` with labels.
#' @param steps Number of steps for which to evaluate model. If `NULL`, evaluates until `input_fn` raises an end-of-input exception.
#' @param hooks List of `SessionRunHook` subclass instances. Used for callbacks inside the evaluation call.
#' @param checkpoint_path Path of a specific checkpoint to evaluate. If `NULL`, the latest checkpoint in `model_dir` is used.
#' @param name Name of the evaluation if user needs to run multiple evaluations on different data sets, such as on training data vs test data. Metrics for different evaluations are saved in separate folders, and appear separately in tensorboard.
#' 
#' @return A dict containing the evaluation metrics specified in `model_fn` keyed by name, as well as an entry `global_step` which contains the value of the global step for which this evaluation was performed.
#' @section Raises:
#' ValueError: If `steps <= 0`. ValueError: If no model has been trained, namely `model_dir`, or the given `checkpoint_path` is empty.
#' 
#' @export
#' @family custom estimator methods
evaluate.tf_custom_model <- function(object,
                                     input_fn,
                                     steps = NULL,
                                     checkpoint_path = NULL,
                                     name = NULL)
{
  validate_custom_model_input_fn(input_fn)
  est <- object$estimator
  est$evaluate(input_fn = input_fn$input_fn,
               steps = as_nullable_integer(steps),
               checkpoint_path = checkpoint_path,
               name = name)
}

#' Exports inference graph as a SavedModel into a given directory.
#' 
#' This method builds a new graph by first calling the
#' serving_input_receiver_fn to obtain feature `Tensor`s, and then calling
#' this `Estimator`'s model_fn to generate the model graph based on those
#' features. It restores the given checkpoint (or, lacking that, the most
#' recent checkpoint) into this graph in a fresh session. Finally it creates
#' a timestamped export directory below the given export_dir_base, and writes
#' a `SavedModel` into it containing a single `MetaGraphDef` saved from this
#' session. The exported `MetaGraphDef` will provide one `SignatureDef` for each
#' element of the export_outputs dict returned from the model_fn, named using
#' the same keys. One of these keys is always
#' signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY, indicating which
#' signature will be served when a serving request does not specify one.
#' For each signature, the outputs are provided by the corresponding
#' `ExportOutput`s, and the inputs are always the input receivers provided by
#' the serving_input_receiver_fn. Extra assets may be written into the SavedModel via the extra_assets
#' argument. This should be a dict, where each key gives a destination path
#' (including the filename) relative to the assets.extra directory. The
#' corresponding value gives the full path of the source file to be copied.
#' For example, the simple case of copying a single file without renaming it
#' is specified as `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
#' 
#' @param export_dir_base A string containing a directory in which to create timestamped subdirectories containing exported SavedModels.
#' @param serving_input_receiver_fn A function that takes no argument and returns a `ServingInputReceiver`.
#' @param assets_extra A dict specifying how to populate the assets.extra directory within the exported SavedModel, or `NULL` if no extra assets are needed.
#' @param as_text whether to write the SavedModel proto in text format.
#' @param checkpoint_path The checkpoint path to export. If `NULL` (the default), the most recent checkpoint found within the model directory is chosen.
#' 
#' @return The string path to the exported directory.
#' 
#' @section Raises:
#' ValueError: if no serving_input_receiver_fn is provided, no export_outputs are provided, or no checkpoint can be found.
#' 
#' @export
#' @family custom estimator methods
export_savedmodel.tf_custom_model <- function(
  object,
  export_dir_base,
  serving_input_receiver_fn,
  assets_extra = NULL,
  as_text = FALSE,
  checkpoint_path = NULL) {
  object$estimator$export_savedmodel(
    export_dir_base = export_dir_base,
    serving_input_receiver_fn = serving_input_receiver_fn,
    assets_extra = assets_extra,
    as_text = as_text,
    checkpoint_path = checkpoint_path
  )
}

#' @export
coef.tf_custom_model <- function(object, ...) {
  coef.tf_model(object, ...)
}


as_model_fn <- function(f) {
  tools <- import_package_module("estimatortools.functions")
  tools$as_model_fn(f)
}



