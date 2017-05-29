tf_model <- function(names, ...) {
  object <- list(...)
  class(object) <- c("tf_model", names)
  object
}

is.tf_model <- function(object) {
  inherits(object, "tf_model")
}

is.classifier <- function(object) {
  inherits(object, "classifier")
}

is.regressor <- function(object) {
  inherits(object, "regressor")
}

#' @export
predict.tf_model <- function(object,
                             input_fn,
                             type = "raw",
                             ...)
{
  validate_input_fn(input_fn)
  est <- object$estimator
  input_fn <- input_fn(get_input_fn_type(object))
  if (is.classifier(object)) {
    if (type == "raw") {
      predictions <- est$predict(input_fn = input_fn, outputs = c("classes"), ...)
    } else if (type == "prob") {
      predictions <- est$predict(input_fn = input_fn, outputs = c("probabilities"), ...)
    } else {
      predictions <- est$predict(input_fn = input_fn, outputs = c(type), ...)
    }
  } else if (is.regressor(object)) {
    if (type == "raw") {
      predictions <- est$predict(input_fn = input_fn, outputs = c("scores"), ...)
    } else {
      predictions <- est$predict(input_fn = input_fn, outputs = c(type), ...)
    }
  } else {
    stop("Right now only classifier and regressor are supported")
  }
  return(unlist(iterate(predictions)))
}

#' @export
train.tf_model <- function(object, input_fn, steps = 2L, monitors = NULL, ...)
{
  validate_input_fn(input_fn)
  if (!is.null(monitors))
    monitors <- list(monitors)
  object$estimator$fit(
    input_fn = input_fn(get_input_fn_type(object)),
    steps = as.integer(steps),
    monitors = monitors,
    ...)
  invisible(object)
}

#' @export
evaluate.tf_model <- function(object, input_fn, steps = 2L, hooks = NULL, ...)
{
  validate_input_fn(input_fn)
  hooks <- normalize_session_run_hooks(hooks)
  object$estimator$evaluate(
    input_fn = input_fn(get_input_fn_type(object)),
    steps = as.integer(steps),
    hooks = hooks,
    ...)
}

#' @importFrom stats coef 
#' @export
coef.tf_model <- function(object, ...) {
  estimator <- object$estimator
  var_names <- estimator$get_variable_names()
  variables <- lapply(var_names, estimator$get_variable_value)
  names(variables) <- var_names
  variables
}

