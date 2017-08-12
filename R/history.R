tf_estimator_history <- function(losses = NULL, steps = NULL) {
  structure(
    list(losses = losses, steps = steps),
    class = "tf_estimator_history"
  )
}

as.data.frame.tf_estimator_history <- function(x, ...) {
  df <- data.frame(
    steps = x$steps,
    losses = x$losses
  )
  rownames(df) <- NULL
  df
}

plot.tf_estimator_history <- function(x, method = c("auto", "ggplot2", "base"), smooth = TRUE) {
  # TODO
}

print.tf_estimator_history <- function(x, ...) {
  print(as.data.frame(x), ...)
}

