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
  
  fields <- list(
    "Model Directory" = estimator$model_dir
  )
  
  body <- enumerate(fields, function(key, val) {
    sprintf("%s: %s", key, val)
  })
  
  output <- paste(
    header,
    body,
    sep = "\n",
    collapse = "\n"
  )
  
  cat(output, sep = "\n")
}
