
#' High-level TF.Learn API in TensorFlow for R
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
#'
#' @docType package
#' @name tflearn
NULL

#' TF.Learn Module
#'
#' \code{learn} acts as an interface to the main \href{https://www.tensorflow.org/tutorials/tflearn/}{TF.Learn}
#' module. Objects and functions defined within this module can be accessed
#' using the \code{$} function.
#'
#' @export
learn <- NULL
contrib_layers <- NULL
feature_column_lib <- NULL
estimator_lib <- NULL
contrib_estimators_lib <- NULL
np <- NULL
random_ops <- NULL
math_ops <- NULL
array_ops <- NULL
functional_ops <- NULL

.onLoad <- function(libname, pkgname) {
  learn <<- reticulate::import("tensorflow.contrib.learn", delay_load = TRUE)
  contrib_layers <<- reticulate::import("tensorflow.contrib.layers", delay_load = TRUE)
  estimator_lib <<- reticulate::import("tensorflow.python.estimator.estimator", delay_load = TRUE)
  contrib_estimators_lib <<- reticulate::import("tensorflow.contrib.learn.python.learn.estimators", delay_load = TRUE)
  feature_column_lib <<- reticulate::import("tensorflow.contrib.layers.python.layers.feature_column", delay_load = TRUE)
  np <<- reticulate::import("numpy", convert = FALSE, delay_load = TRUE)
  random_ops <<- reticulate::import("tensorflow.python.ops.random_ops", delay_load = TRUE)
  math_ops <<- reticulate::import("tensorflow.python.ops.math_ops", delay_load = TRUE)
  array_ops <<- reticulate::import("tensorflow.python.ops.array_ops", delay_load = TRUE)
  functional_ops <<- reticulate::import("tensorflow.python.ops.functional_ops", delay_load = TRUE)
}

.onUnload <- function(libpath) {

}

.onAttach <- function(libname, pkgname) {

}

.onDetach <- function(libpath) {

}
