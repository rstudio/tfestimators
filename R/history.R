new_tf_estimator_history <- function(losses = NULL, steps = NULL) {
  metrics <- names(losses)
  steps <- tail(steps, 1)
  structure(
    list(
      params = list(metrics = metrics,
                    steps = steps),
      losses = losses, 
      step = steps
      ),
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

