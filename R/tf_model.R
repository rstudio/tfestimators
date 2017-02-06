tf_model <- function(name, ...) {
  object <- list(...)
  class(object) <- c("tf_model", sprintf("tf_model_%s", name))
  object
}

is.tf_model <- function(object) {
  inherits(object, "tf_model")
}

#' @export
predict.tf_model <- function(object,
                             newdata = NULL,
                             input_fn = NULL,
                             type = "raw",
                             ...)
{
  est <- object$estimator
  default_input_fn <- object$recipe$input_fn
  if(is.null(input_fn) && is.null(newdata)) {
    warning("Neither input_fn or newdata is provided, using the same input_fn specified in recipe")
    input_fn <- default_input_fn
  }
  if(!is.null(newdata)) {
    input_fn <- function(){default_input_fn(newdata = newdata)}
  }
  if(type == "raw") {
    predictions <- est$predict(input_fn = input_fn, ...)
  } else if (type == "prob") {
    # this only works for classification problems
    if(length(grep("classification", class(object))) == 0) {
      stop("type = prob only works for classification problems")
    }
    predictions <- est$predict_proba(input_fn = input_fn, ...)
  } else {
    stop(paste0("This type is not supported: ", as.character(type)))
  }
  return(unlist(iterate(predictions)))
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
