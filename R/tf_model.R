tf_model <- function(names, ...) {
  simple_class(c("tf_model", names), ...)
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
  input_fn <- normalize_input_fn(object, input_fn)
  est <- object$estimator
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
  monitors <- normalize_session_run_hooks(monitors)
  object$estimator$fit(
    input_fn = normalize_input_fn(object, input_fn),
    steps = as.integer(steps),
    monitors = monitors,
    ...)
  invisible(object)
}


#' @export
evaluate.tf_model <- function(object, input_fn, steps = 2L, hooks = NULL, ...)
{
  hooks <- normalize_session_run_hooks(hooks)
  object$estimator$evaluate(
    input_fn = normalize_input_fn(object, input_fn),
    steps = as.integer(steps),
    hooks = hooks,
    ...)
}

#' @importFrom stats coef 
#' @export
coef.tf_model <- function(object) {
  coef.tf_custom_model(object)
}

