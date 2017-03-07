tf_custom_model <- function(...) {
  object <- list(...)
  class(object) <- "tf_custom_model"
  object
}

is.tf_custom_model <- function(object) {
  inherits(object, "tf_custom_model")
}

#' @export
estimator_spec <- function(predictions,
                           loss,
                           train_op,
                           mode = "train",
                           ...) {
  learn$ModelFnOps(
    mode = mode,
    predictions = predictions,
    loss = loss,
    train_op = train_op,
    ...)
}

#' @export
estimator <- function(model_fn,
                      run_options = NULL,
                      ...)
{
  run_options <- run_options %||% run_options()

  est <- learn$Estimator(
    model_fn = model_fn,
    model_dir = run_options$model_dir,
    config = run_options$run_config,
    ...
  )
  tf_custom_model(estimator = est, model_fn = model_fn)
}

#' @export
fit.tf_custom_model <- function(object, input_fn, ...) {
  fit.tf_model(object, input_fn = input_fn, ...)
}

#' @export
predict.tf_custom_model <- function(object,
                                    input_fn = NULL,
                                    type = "raw",
                                    ...) {
  est <- object$estimator
  predictions <- est$predict(input_fn = input_fn, ...) %>% iterate
  if (length(names(predictions)) == 1) {
    # regression
    return(predictions)
  } else {
    # classification
    if (type == "raw") {
      unlist(lapply(predictions, function(prediction){
        prediction$class
      }))
    } else if (type == "prob") {
      unlist(lapply(predictions, function(prediction){
        prediction$prob
      }))
    } else {
      stop(paste0("This type is not supported for classification problem: ", as.character(type)))
    }
  }
}

#' @export
coef.tf_custom_model <- function(object, ...) {
  coef.tf_model(object, ...)
}
