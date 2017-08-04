tf_estimator_type <- function(estimator) {
  if (inherits(estimator, "tf_estimator_regressor"))
    "regressor"
  else if (inherits(estimator, "tf_estimator_classifier"))
    "classifier"
  else
    "estimator"
}

#' @export
print.tf_estimator <- function(x, ...) {
  
  header <- sprintf(
    "A TensorFlow %s [%s]",
    tf_estimator_type(x),
    as.character(x$estimator)
  )
  
  model_dir <-  x$estimator$model_dir

  fields <- list(
    "Model Directory" = model_dir
  )
  
  body <- enumerate(fields, function(key, val) {
    sprintf("%s: %s", key, val)
  })

  # Model checkpoint only exists when it's been trained
  if (dir.exists(model_dir)) {
    global_step <- coef(x)[[graph_keys()$GLOBAL_STEP]]
    model_trained_info <- sprintf(
      "Model has been trained for %i %s.",
      as.integer(global_step),
      if (global_step > 1) "steps" else "step"
    )
  } else {
    model_trained_info <- sprintf("Model has not yet been trained.")
  }

  output <- paste(
    header,
    body,
    model_trained_info,
    sep = "\n",
    collapse = "\n"
  )
  
  cat(output, sep = "\n")
}
