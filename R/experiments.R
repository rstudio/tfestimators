attach_data_to_input_fn <- function(dt, input_fn) {
  if (is.null(dt))
    input_fn
  else
    function() {
      input_fn(newdata = dt)
    }
}

#' @export
setup_experiment <- function(tf_model,
                             train_data,
                             eval_data,
                             train_steps = 2L,
                             eval_steps = 2L,
                             ...) {

  if(!is.tf_model(tf_model)) stop("tf_model must be a tf_model object")
  not_allowed_args <- c("train_input_fn", "eval_input_fn", "estimator")
  addtional_args <- list(...)
  if(length(addtional_args) != 0 && (names(addtional_args) %in% not_allowed_args)) {
    stop("You cannot use the following args: ", paste(not_allowed_args, collapse = ", "))
  }

  default_input_fn <- tf_model$recipe$input_fn
  train_input_fn <- attach_data_to_input_fn(train_data, default_input_fn)
  eval_input_fn <- attach_data_to_input_fn(eval_data, default_input_fn)
  
  tf$contrib$learn$Experiment(estimator = tf_model$estimator,
                              train_input_fn = train_input_fn,
                              eval_input_fn = eval_input_fn,
                              train_steps = train_steps,
                              eval_steps = eval_steps,
                              ...)
}
