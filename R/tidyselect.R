tidyselect_data <- function() {
  tidyselect <- asNamespace("tidyselect")
  exports <- getNamespaceExports(tidyselect)
  mget(exports, tidyselect, inherits = TRUE)
}

#' Establish a Feature Columns Selection Scope
#' 
#' This helper function provides a set of names to be
#' used by `tidyselect` helpers in e.g. [feature_columns()].
#' 
#' @param columns Either a named \R object (whose names will be
#'   used to provide a selection context), or a character vector
#'   of such names.
#'   
#' @param expr An \R expression, to be evaluated with the selection
#'   context active.
#'
#' @name column-scope
NULL

#' @rdname column-scope
#' @export
set_columns <- function(columns) {
  tidyselect::poke_vars(object_names(columns))
}

#' @rdname column-scope
#' @export
with_columns <- function(columns, expr) {
  tidyselect::with_vars(object_names(columns), expr)
}

#' @rdname column-scope
#' @export
scoped_columns <- function(columns) {
  tidyselect::scoped_vars(object_names(columns), parent.frame())
}
