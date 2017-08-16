pad <- function(object, n) {
  UseMethod("pad")
}

#' @export
pad.default <- function(object, n) {
  if (n - length(object) <= 0)
    return(object)
  
  c(object, rep.int(NA, n - length(object)))
}

#' @export
pad.data.frame <- function(object, n) {
  
  # compute padding (bail if nothing to do)
  padding <- n - nrow(object)
  if (padding <= 0)
    return(object)
  
  # reset row names to new length
  nrow <- nrow(object)
  attr(object, "row.names") <- c(NA_integer_, -as.integer(n))
  
  # extend vectors in object
  for (i in seq_along(object))
    object[[i]] <- c(object[[i]], rep(NA, times = padding))
  
  # and we're done!
  object
}
