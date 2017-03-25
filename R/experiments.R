#' @export
tf_experiment <- function(name, ...) {
  object <- list(...)
  class(object) <- "tf_experiment"
  object
}

#' @export
experiment <- function(x, ...) {
  UseMethod("experiment")
}

#' @export
experiment.tf_custom_model <- function(object, ...) {
  experiment.tf_model(object, ...)
}

#' @export
train_and_evaluate.tf_experiment <- function(object) {
  object$experiment$train_and_evaluate()
}


#' @export
evaluate.tf_experiment <- function(object, delay_secs = NULL) {
  object$experiment$evaluate(delay_secs = delay_secs)
}


#' @export
train.tf_experiment <- function(object, delay_secs = NULL) {
  object$experiment$train(delay_secs = delay_secs)
}

#' @export
experiment.tf_model <- function(object,
                                train_input_fn,
                                eval_input_fn,
                                train_steps = 2L,
                                eval_steps = 2L,
                                ...) {
  
  exp <- tf$contrib$learn$Experiment(
    estimator = object$estimator,
    train_input_fn = train_input_fn$input_fn,
    eval_input_fn = eval_input_fn$input_fn,
    train_steps = train_steps,
    eval_steps = eval_steps,
    ...)
  tf_experiment(experiment = exp)
}
