new_tf_keras_estimator <- function(estimator, args = NULL, ...,
                                   subclass = NULL) {
  new_tf_estimator(estimator, args, ...,
                   subclass = c(subclass, "tf_keras_estimator"))
}

#' Keras Estimators
#'
#' Create an Estimator from a compiled Keras model
#' 
#' @param keras_model A keras model.
#' @param keras_model_path Directory to a keras model on disk.
#' @param custom_objects Dictionary for custom objects.
#' @param model_dir Directory to save Estimator model parameters, graph and etc.
#' @param config Configuration object.
#'
#' @export
keras_model_to_estimator <- function(
  keras_model = NULL, keras_model_path = NULL, custom_objects = NULL,
  model_dir = NULL, config = NULL) {
  
  if (is.null(keras_model) && is.null(keras_model_path))
    stop("Either keras_model or keras_model_path needs to be provided.")
  
  if (!is.null(keras_model_path)) {
    if (!is.null(keras_model))
      stop("Please specity either keras_model or keras_model_path but not both.")
    if (grepl("^(gs://|storage\\.googleapis\\.com)", keras_model_path))
      stop("'keras_model_path' is not a local path. Please copy the model locally first.")
    keras_model <- tf$keras$models$load_model(keras_model_path)
  }
  
  tryCatch(reticulate::py_get_attr(keras_model, "optimizer"),
           error = function(e) stop(
             "Given keras model has not been compiled yet. Please compile first\n
             before creating the estimator.")
           )
  
  args <- as.list(environment(), all = TRUE)
  
  estimator <- py_suppress_warnings(
    tf$keras$estimator$model_to_estimator(
      keras_model = keras_model,
      keras_model_path = keras_model_path,
      custom_objects = custom_objects,
      model_dir = model_dir,
      config = config
    ))
  
  new_tf_keras_estimator(estimator, args = args)
}
