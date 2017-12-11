tf_estimator_history <- function(losses = NULL, steps = NULL) {
  structure(
    list(losses = losses, steps = steps),
    class = "tf_estimator_history"
  )
}

#' @export
as.data.frame.tf_estimator_history <- function(x, ...) {
    data.frame(x[["losses"]]) %>%
      cbind(data.frame(x["steps"])) %>%
      tidyr::gather("metric", "value", -"steps")
}

#' @export
print.tf_estimator_history <- function(x, ...) {
  print(as.data.frame(x), ...)
}

