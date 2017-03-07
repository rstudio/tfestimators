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
                             input_fn = NULL,
                             type = "raw",
                             ...)
{
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
fit <- function(object, ...) {
  UseMethod("fit")
}

#' @export
fit.tf_model <- function(object, input_fn = NULL, steps = 2L, monitors = NULL, ...)
{
  if (!is.null(monitors))
    monitors <- list(monitors)
  object$estimator$fit(
    input_fn = input_fn,
    steps = steps,
    monitors = monitors,
    ...)
  object
}

#' @export
coef.tf_model <- function(object, ...) {
  estimator <- object$estimator
  var_names <- estimator$get_variable_names()
  variables <- lapply(var_names, estimator$get_variable_value)
  names(variables) <- var_names
  variables
}

#' @export
summary.tf_model <- function(object, ...) {
  # NYI
}
