attach_data_to_input_fn <- function(dt, input_fn) {
  if (is.null(dt))
    input_fn
  else
    function() {
      input_fn(newdata = dt)
    }
}

#' @export
tf_experiment <- function(name, ...) {
  object <- list(...)
  class(object) <- "tf_experiment"
  object
}

#' @export
setup_experiment <- function(x, ...) {
  UseMethod("setup_experiment")
}

#' @export
setup_experiment.tf_custom_model <- function(object, ...) {
  setup_experiment.tf_model(object, ...)
}

#' @export
train_and_evaluate <- function(x, ...) {
  UseMethod("train_and_evaluate")
}

#' @export
train_and_evaluate.tf_experiment <- function(object, ...) {
  object$experiment$train_and_evaluate(...)
}

#' @export
setup_experiment.tf_model <- function(object,
                                      train_data,
                                      eval_data,
                                      train_steps = 2L,
                                      eval_steps = 2L,
                                      ...) {

  default_input_fn <- object$recipe$input_fn
  train_input_fn <- attach_data_to_input_fn(train_data, default_input_fn)
  eval_input_fn <- attach_data_to_input_fn(eval_data, default_input_fn)
  
  exp <- tf$contrib$learn$Experiment(
    estimator = object$estimator,
    train_input_fn = train_input_fn,
    eval_input_fn = eval_input_fn,
    train_steps = train_steps,
    eval_steps = eval_steps,
    ...)
  tf_experiment(experiment = exp)
}
