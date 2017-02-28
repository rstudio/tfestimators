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

#' @export
predict.tf_model <- function(object,
                             input_fn = NULL,
                             type = "raw",
                             ...)
{
  est <- object$estimator
  if (type == "raw") {
    predictions <- est$predict(input_fn = input_fn, ...)
  } else if (type == "prob") {
    # this only works for classification problems
    if (!is.classifier(object)) {
      stop("type = prob only works for classification problems")
    }
    predictions <- est$predict_proba(input_fn = input_fn, ...)
  } else {
    stop(paste0("This type is not supported: ", as.character(type)))
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
