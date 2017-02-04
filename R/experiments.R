attach_data_to_input_fn <- function(dt, input_fn) {
  if(is.null(dt)) {
    return(input_fn)
  } else {
    return(function(){input_fn(newdata = dt)})
  }
}

#' @export
setup_experiment <- function(tf_model,
                             train_data = NULL,
                             eval_data = NULL,
                             train_steps = 2L,
                             eval_steps = 2L,
                             ...) {
  # TODO: Check tf_model class
  # TODO: Check edge cases
  default_input_fn <- tf_model$recipe$input.fn
  train_input_fn <- attach_data_to_input_fn(train_data, default_input_fn)
  eval_input_fn <- attach_data_to_input_fn(eval_data, default_input_fn)
  
  tf$contrib$learn$Experiment(estimator = tf_model$estimator,
                              train_input_fn = train_input_fn,
                              eval_input_fn = eval_input_fn,
                              train_steps = train_steps,
                              eval_steps = eval_steps,
                              ...)
}
