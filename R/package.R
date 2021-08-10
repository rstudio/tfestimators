#' High-level Estimator API in TensorFlow for R
#' 
#' This library provides an R interface to the
#' \href{https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/estimator}{Estimator}
#' API inside TensorFlow that's designed to streamline the process of creating,
#' evaluating, and deploying general machine learning and deep learning models.
#' 
#' \href{https://www.tensorflow.org}{TensorFlow} is an open source software library 
#' for numerical computation using data flow graphs. Nodes in the graph 
#' represent mathematical operations, while the graph edges represent the 
#' multidimensional data arrays (tensors) communicated between them. The 
#' flexible architecture allows you to deploy computation to one or more CPUs or
#' GPUs in a desktop, server, or mobile device with a single API.
#' 
#' The \href{https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/all_symbols}{TensorFlow 
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
      check_tensorflow_version(displayed_warning)
    },
    
    on_error = function(e) {
      stop(tf_config()$error_message, call. = FALSE)
    }
  )
  
 
  # core modules
  if (package_version(Sys.getenv("TENSORFLOW_VERSION", "1.15")) <= "1.12") {
    estimator_lib <<- import("tensorflow.python.estimator.estimator", delay_load = delay_load)
    feature_column_lib <<- import("tensorflow.python.feature_column.feature_column", delay_load = delay_load)
    canned_estimator_lib <<- import("tensorflow.python.estimator.canned", delay_load = delay_load)
  } else {
    estimator_lib <<- import("tensorflow_estimator.python.estimator.estimator", delay_load = delay_load)
    feature_column_lib <<- import("tensorflow.python.feature_column.feature_column_v2", delay_load = delay_load)
    canned_estimator_lib <<- import("tensorflow_estimator.python.estimator.canned", delay_load = delay_load)
  }
  

  # other modules
  np <<- import("numpy", convert = FALSE, delay_load = TRUE)
}

check_tensorflow_version <- function(displayed_warning) {
  current_tf_ver <- tf_version()
  required_least_ver <- "1.3"
  if (current_tf_ver < required_least_ver) {
    if (!displayed_warning) {
      message("tfestimators requires TensorFlow version >= ", required_least_ver, " ",
              "(you are currently running version ", current_tf_ver, ").\n")
      displayed_warning <<- TRUE
    }
  }
}

.onUnload <- function(libpath) {

}

.onAttach <- function(libname, pkgname) {
  msg <- "tfestimators is not recomended for new code. It is only compatible with Tensorflow version 1, and is not compatable with Tensorflow version 2."
  packageStartupMessage(msg)
}

.onDetach <- function(libpath) {

}
