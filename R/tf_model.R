tf_model <- function(name, ...) {
  object <- list(...)
  class(object) <- c("tf_model", sprintf("tf_model_%s", name))
  object
}

#' @importFrom stats predict
#' @export
predict.tf_model <- function(object, newdata, ...) {
  # NYI
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
