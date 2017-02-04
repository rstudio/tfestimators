#' @export
setup_experiment <- function(estimator,
                             train_input_fn,
                             eval_input_fn,
                             train_steps,
                             eval_steps,
                             ...) {
  tf$contrib$learn$Experiment(estimator = estimator,
                              train_input_fn = train_input_fn,
                              eval_input_fn = eval_input_fn,
                              train_steps = train_steps,
                              eval_steps = eval_steps,
                              ...)
}
