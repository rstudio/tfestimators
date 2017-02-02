tf_model <- function(name, ...) {
  object <- list(...)
  class(object) <- c("tf_model", sprintf("tf_model_%s", name))
  object
}

#' @importFrom stats predict
#' @export
predict.tf_model <- function(object, input_fn = NULL, type = "raw", ...) {
  est <- object$estimator
  if(is.null(input_fn)) {
    warning("input_fn is not provided, using the same input_fn specified in recipe")
    input_fn <- object$recipe$input.fn
  }
  if(type == "raw") {
    predictions <- est$predict(input_fn = input_fn, ...)
  } else if (type == "prob") {
    # this only works for classification problems
    if(length(grep("classification", class(reg))) == 0) {
      stop("type = prob only works for classification problems")
    }
    predictions <- est$predict_proba(input_fn = input_fn, ...)
  } else {
    stop(paste0("This type is not supported: ", as.character(type)))
  }
  return(unlist(iterate(predictions)))
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

#' @export
summary.tf_model <- function(object, ...) {
  # NYI
}
