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
#' \href{https://rstudio.github.io/tensorflow}{https://rstudio.github.io/tensorflow}
#' 
#' @import reticulate
#' @import tensorflow
#'   
#' @docType package
#' @name tfestimators
NULL

estimator_lib <- NULL
random_ops <- NULL
math_ops <- NULL
array_ops <- NULL
functional_ops <- NULL

contrib_learn <- NULL
contrib_layers <- NULL
contrib_feature_column_lib <- NULL
contrib_estimators_lib <- NULL

np <- NULL

.onLoad <- function(libname, pkgname) {
  # core modules
  estimator_lib <<- import("tensorflow.python.estimator.estimator", delay_load = TRUE)
  feature_column_lib <<- import("tensorflow.python.feature_column.feature_column", delay_load = TRUE)
  random_ops <<- import("tensorflow.python.ops.random_ops", delay_load = TRUE)
  math_ops <<- import("tensorflow.python.ops.math_ops", delay_load = TRUE)
  array_ops <<- import("tensorflow.python.ops.array_ops", delay_load = TRUE)
  functional_ops <<- import("tensorflow.python.ops.functional_ops", delay_load = TRUE)

  # contrib modules
  contrib_learn <<- import("tensorflow.contrib.learn", delay_load = TRUE)
  contrib_layers <<- import("tensorflow.contrib.layers", delay_load = TRUE)
  contrib_estimators_lib <<- import("tensorflow.contrib.learn.python.learn.estimators", delay_load = TRUE)

  # other modules
  np <<- import("numpy", convert = FALSE, delay_load = TRUE)
}

.onUnload <- function(libpath) {

}

.onAttach <- function(libname, pkgname) {

}

.onDetach <- function(libpath) {

}
