tf_custom_model <- function(...) {
  object <- list(...)
  class(object) <- "tf_custom_model"
  object
}

is.tf_custom_model <- function(object) {
  inherits(object, "tf_custom_model")
}

#' @export
custom_model_return_fn <- function(logits, loss, train_op, mode = "train") {
  learn$ModelFnOps(
    mode = mode,
    predictions = list(
      class = tf$argmax(logits, 1L),
      prob = tf$nn$softmax(logits)),
    loss = loss,
    train_op = train_op)
}

#' @export
create_custom_estimator <- function(model_fn, input_fn, steps, model_dir, config, ...) {
  est <- tf$contrib$learn$Estimator(
    model_fn = model_fn,
    model_dir = model_dir,
    config = config,
    ...)
  est$fit(input_fn = input_fn, steps = steps)
  tf_custom_model(estimator = est, model_fn = model_fn)
}

#' @export
predict.tf_custom_model <- function(object,
                                    newdata = NULL,
                                    input_fn = NULL,
                                    type = "raw",
                                    ...) {
  est <- object$estimator
  input_fn <- prepare_predict_input_fn(object, newdata, input_fn)
  predictions <- est$predict(input_fn = input_fn, ...) %>% iterate
  if (type == "raw") {
    unlist(lapply(predictions, function(prediction){
      prediction$class
    }))
  } else if (type == "prob") {
    unlist(lapply(predictions, function(prediction){
      prediction$prob
    }))
  } else {
    stop(paste0("This type is not supported: ", as.character(type)))
  }
}
