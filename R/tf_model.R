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

prepare_input_fn <- function(object,
                             newdata = NULL,
                             input_fn = NULL) {
  est <- object$estimator
  default_input_fn <- object$recipe$input_fn
  if (is.null(input_fn) && is.null(newdata)) {
    warning("Neither input_fn or newdata is provided, using the same input_fn specified in recipe")
    return(default_input_fn)
  } else if (!is.null(newdata)) {
    return(function(){default_input_fn(newdata = newdata)})
  } else {
    return(input_fn)
  }
}

#' @export
predict.tf_model <- function(object,
                             newdata = NULL,
                             input_fn = NULL,
                             type = "raw",
                             ...)
{
  est <- object$estimator
  prepared_input_fn <- prepare_input_fn(object, newdata, input_fn)
  if (type == "raw") {
    predictions <- est$predict(input_fn = prepared_input_fn, ...)
  } else if (type == "prob") {
    # this only works for classification problems
    if (!is.classifier(object)) {
      stop("type = prob only works for classification problems")
    }
    predictions <- est$predict_proba(input_fn = prepared_input_fn, ...)
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
fit.tf_model <- function(object, data = NULL, input_fn = NULL, steps = 2L, monitors = NULL, ...)
{
  if (!is.null(monitors))
    monitors <- list(monitors)
  suppressWarnings(prepared_input_fn <- prepare_input_fn(object, data, input_fn))
  object$estimator$fit(
    input_fn = prepared_input_fn,
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
