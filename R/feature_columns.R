contrib_layers <- tf$contrib$layers

#' @export
column_with_keys <- function(...) {
  contrib_layers$sparse_column_with_keys(...)
}

#' @export
column_with_hash_bucket <- function(...) {
  contrib_layers$sparse_column_with_hash_bucket(...)
}

#' @export
column_real_valued <- function(...) {
  contrib_layers$real_valued_column(...)
}

#' @export
column_embedding <- function(...) {
  contrib_layers$embedding_column(...)
}

#' @export
column_crossed <- function(...) {
  contrib_layers$crossed_column(...)
}

