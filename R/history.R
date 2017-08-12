tf_estimator_history <- function(losses = NULL, steps = NULL) {
  structure(
    list(losses = losses, steps = steps),
    class = "tf_estimator_history"
  )
}

#' @export
as.data.frame.tf_estimator_history <- function(x, ...) {
  df <- cbind(
    as.data.frame(x$losses),
    data.frame(steps = x$steps))
  rownames(df) <- NULL
  df
}

plot.tf_estimator_history <- function(x, method = c("auto", "ggplot2", "base"), smooth = TRUE) {
  # TODO
}

#' @export
print.tf_estimator_history <- function(x, ...) {
  print(as.data.frame(x), ...)
}

