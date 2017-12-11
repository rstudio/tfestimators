new_tf_estimator_history <- function(losses = NULL, step = NULL) {
  metrics <- names(losses)
  steps <- tail(step, 1)
  structure(
    list(
      params = list(metrics = metrics,
                    steps = steps),
      losses = losses, 
      step = step
      ),
    class = "tf_estimator_history"
  )
}

#' @export
as.data.frame.tf_estimator_history <- function(x, ...) {
    data.frame(x[["losses"]]) %>%
      cbind(data.frame(x["step"])) %>%
      tidyr::gather("metric", "value", -"step")
}

#' @export
print.tf_estimator_history <- function(x, ...) {
  print(as.data.frame(x), ...)
}

