#' Train a Model
#' 
#' Train a model object.
#' 
#' @param object A trainable \R object.
#' @param ... Optional arguments passed on to implementing methods.
#' 
#' @export
train <- function(object, ...) {
  UseMethod("train")
}

#' Evaluate a Model
#' 
#' Evaluate a model object.
#' 
#' @param object An evaluatable \R object.
#' @param ... Optional arguments passed on to implementing methods.
#' 
#' @export
evaluate <- function(object, ...) {
  UseMethod("evaluate")
}

#' Simultaneously Train and Evaluate a Model
#' 
#' Train and evaluate a model object.
#' 
#' @param object An \R object.
#' @param ... Optional arguments passed on to implementing methods.
#' 
#' @export
train_and_evaluate <- function(object, ...) {
  UseMethod("train_and_evaluate")
}

#' Construct an Experiment
#' 
#' Construct an experiment object.
#' 
#' @param object An \R object.
#' @param ... Optional arguments passed on to implementing methods.
experiment <- function(object, ...) {
  UseMethod("experiment")
}


