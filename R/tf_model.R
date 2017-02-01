tf_model <- function(name, ...) {
  object <- list(...)
  class(object) <- c("tf_model", sprintf("tf_model_%s", name))
  object
}

#' @importFrom stats predict
#' @export
predict.tf_model <- function(object, newdata, type = "raw", ...) {
  # est <- object$estimator
  # if(type == "raw") {
  #   est$predict(input_fn, batch_size)
  # } else if (type == "prob") {
  #   est$predict_proba(input_fn, batch_size)
  # } else {
  #   stop(paste0("This type is not supported: ", as.character(type)))
  # }
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
