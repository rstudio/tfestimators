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

#' @export
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

#' @export
estimator <- function(model_fn,
                      run_options = NULL,
                      ...)
{
  run_options <- run_options %||% run_options()

  model_fn <- as_model_fn(model_fn)
  est <- estimator_lib$Estimator(
    model_fn = model_fn,
    model_dir = run_options$model_dir,
    ...
  )
  tf_custom_model(estimator = est, model_fn = model_fn)
}

#' @export
fit.tf_custom_model <- function(object, input_fn, steps = 2L, ...) {
  validate_custom_model_input_fn(input_fn)
  object$estimator$train(input_fn = input_fn$input_fn, steps = steps, ...)
  object
}

#' @export
predict.tf_custom_model <- function(object,
                                    input_fn,
                                    as_vector = T,
                                    checkpoint_path = NULL,
                                    ...) {
  validate_custom_model_input_fn(input_fn)
  est <- object$estimator
  predictions <- est$predict(
    input_fn = input_fn$input_fn,
    checkpoint_path = checkpoint_path,
    ...)
  if (as_vector) {
    if (!any(inherits(predictions, "python.builtin.iterator"),
             inherits(predictions, "python.builtin.generator"))) {
      warning("predictions are not iterable, no need to convert again")
    } else {
      predictions <- predictions %>% iterate
    }
    if (is.list(predictions)) predictions <- unlist(predictions)
  }
  predictions
}

#' @export
evaluate.tf_custom_model <- function(object,
                                     input_fn,
                                     steps = 2L,
                                     checkpoint_path = NULL,
                                     ...)
{
  validate_custom_model_input_fn(input_fn)
  est <- object$estimator
  est$evaluate(input_fn = input_fn$input_fn,
               steps = steps,
               checkpoint_path = checkpoint_path,
               ...)
}

#' @export
get_latest_checkpoint <- function(checkpoint_dir, ...) {
  if (!dir.exists(checkpoint_dir)) {
    stop(paste0("This checkpoint_dir does not exist: ", checkpoint_dir))
  }
  tf$python$training$saver$latest_checkpoint(checkpoint_dir, ...) 
}

#' @export
coef.tf_custom_model <- function(object, ...) {
  coef.tf_model(object, ...)
}


as_model_fn <- function(f) {
  tools <- import_package_module("tflearntools.functions")
  tools$as_model_fn(f)
}



