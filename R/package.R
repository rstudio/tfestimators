#' High-level Estimator API in TensorFlow for R
#' 
#' This library provides an R interface to the
#' \href{https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/estimator}{Estimator}
#' API inside TensorFlow that's designed to streamline the process of creating,
#' evaluating, and deploying general machine learning and deep learning models.
#' 
#' \href{https://tensorflow.org}{TensorFlow} is an open source software library 
#' for numerical computation using data flow graphs. Nodes in the graph 
#' represent mathematical operations, while the graph edges represent the 
#' multidimensional data arrays (tensors) communicated between them. The 
#' flexible architecture allows you to deploy computation to one or more CPUs or
#' GPUs in a desktop, server, or mobile device with a single API.
#' 
#' The \href{https://www.tensorflow.org/api_docs/python/index.html}{TensorFlow 
#' API} is composed of a set of Python modules that enable constructing and 
#' executing TensorFlow graphs. The tensorflow package provides access to the 
#' complete TensorFlow API from within R.
#' 
#' For additional documentation on the tensorflow package see 
#' \href{https://tensorflow.rstudio.com}{https://tensorflow.rstudio.com}
#' 
#' @docType package
#' @name tfestimators
NULL

estimator_lib <- NULL
feature_column_lib <- NULL
canned_estimator_lib <- NULL
random_ops <- NULL
math_ops <- NULL
array_ops <- NULL
functional_ops <- NULL

contrib_learn <- NULL
contrib_layers <- NULL
contrib_estimators_lib <- NULL

np <- NULL

.globals <- new.env(parent = emptyenv())
.globals$active_column_names <- NULL
.globals$history <- NULL

.onLoad <- function(libname, pkgname) {
  
  # delay load handler
  displayed_warning <- FALSE
  delay_load <- list(
    
    priority = 5,
    
    environment = "r-tensorflow",
    
    on_load = function() {
      current_tf_ver <- tf_version()
      required_least_ver <- "1.3"
      if (current_tf_ver < required_least_ver) {
        if (!displayed_warning) {
          message("tfestimators requires TensorFlow version > ", required_least_ver, " ",
                  "(you are currently running version ", current_tf_ver, ").\n")
          displayed_warning <<- TRUE
        }
      }
    },
    
    on_error = function(e) {
      stop(tf_config()$error_message, call. = FALSE)
    }
  )
  
  # core modules
  estimator_lib <<- import("tensorflow.python.estimator.estimator", delay_load = delay_load)
  feature_column_lib <<- import("tensorflow.python.feature_column.feature_column", delay_load = delay_load)
  random_ops <<- import("tensorflow.python.ops.random_ops", delay_load = delay_load)
  math_ops <<- import("tensorflow.python.ops.math_ops", delay_load = delay_load)
  array_ops <<- import("tensorflow.python.ops.array_ops", delay_load = delay_load)
  functional_ops <<- import("tensorflow.python.ops.functional_ops", delay_load = delay_load)
  canned_estimator_lib <<- import("tensorflow.python.estimator.canned", delay_load = delay_load)

  # contrib modules
  contrib_learn <<- import("tensorflow.contrib.learn", delay_load = delay_load)
  contrib_layers <<- import("tensorflow.contrib.layers", delay_load = delay_load)
  contrib_estimators_lib <<- import("tensorflow.contrib.learn.python.learn.estimators", delay_load = delay_load)

  # other modules
  np <<- import("numpy", convert = FALSE, delay_load = TRUE)
}

.onUnload <- function(libpath) {

}

.onAttach <- function(libname, pkgname) {

}

.onDetach <- function(libpath) {

}
