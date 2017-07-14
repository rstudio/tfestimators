tf_model <- function(names, ...) {
  object <- list(...)
  class(object) <- c("tf_model", names)
  object
}

is.tf_model <- function(object) {
  inherits(object, "tf_model")
}

is.classifier <- function(object) {
  inherits(object, "classifier")
}

is.regressor <- function(object) {
  inherits(object, "regressor")
}

#' @export
#' @inherit predict.tf_custom_model
predict.tf_model <- function(object,
                             input_fn,
                             checkpoint_path = NULL,
                             predict_keys = NULL,
                             hooks = NULL,
                             as_iterable = FALSE)
{
  predict.tf_custom_model(object,
                          input_fn,
                          checkpoint_path = checkpoint_path,
                          predict_keys = predict_keys,
                          hooks = hooks,
                          as_iterable = as_iterable)
}


#' @export
#' @inherit train.tf_custom_model
train.tf_model <- function(object,
                           input_fn,
                           steps = NULL,
                           max_steps = NULL,
                           hooks = NULL)
{
  train.tf_custom_model(
    object,
    input_fn = input_fn,
    steps = steps,
    max_steps = max_steps,
    hooks = hooks)
}


#' @export
#' @inherit evaluate.tf_custom_model
evaluate.tf_model <- function(object,
                              input_fn,
                              steps = NULL,
                              checkpoint_path = NULL,
                              name = NULL,
                              hooks = NULL)
{
  evaluate.tf_custom_model(
    object,
    input_fn = input_fn,
    steps = steps,
    checkpoint_path = checkpoint_path,
    name = name,
    hooks = hooks)
}

#' @importFrom stats coef 
#' @export
#' @inherit coef.tf_custom_model
coef.tf_model <- function(object) {
  coef.tf_custom_model(object)
}

