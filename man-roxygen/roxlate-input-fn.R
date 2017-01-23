#' @param input.fn An \R function accepting an input dataset and returning a
#'   two-element list -- the first element itself being an \R list mapping
#'   features (by name) to tensors, and the second defining the response.
#'   When \code{NULL}, the input function will be automatically constructed
#'   according to the designated model inputs.
