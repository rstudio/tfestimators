

#' @importFrom tensorflow evaluate train train_and_evaluate export_savedmodel
NULL


#' Construct an Experiment
#' 
#' Construct an experiment object.
#' 
#' @param object An \R object.
#' @param ... Optional arguments passed on to implementing methods.
experiment <- function(object, ...) {
  UseMethod("experiment")
}


