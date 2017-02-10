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
create_custom_estimator <- function(recipe,
                                    run_options = NULL,
                                    skip_fit = FALSE, ...)
{
  run_options <- run_options %||% run_options()

  est <- tf$contrib$learn$Estimator(
    model_fn = recipe$model_fn,
    model_dir = run_options$model_dir,
    config = run_options$run_config,
    ...
  )
  if (!skip_fit)
    est$fit(input_fn = recipe$input_fn, steps = run_options$steps)
  tf_custom_model(estimator = est, recipe = recipe)
}

#' @export
predict.tf_custom_model <- function(object,
                                    newdata = NULL,
                                    input_fn = NULL,
                                    type = "raw",
                                    ...) {
  est <- object$estimator
  input_fn <- prepare_predict_input_fn(object, newdata, object$recipe$input_fn)
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
