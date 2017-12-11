"%||%" <- function(x, y) if (is.null(x)) y else x

stopf <- function(fmt, ..., call. = TRUE, domain = NULL) {
  stop(simpleError(
    sprintf(fmt, ...),
    if (call.) sys.call(sys.parent())
  ))
}

warnf <- function(fmt, ..., call. = TRUE, immediate. = FALSE) {
  warning(sprintf(fmt, ...), call. = call., immediate. = immediate.)
}

enumerate <- function(object, f, ...) {
  nm <- names(object)
  result <- lapply(seq_along(object), function(i) {
    f(nm[[i]], object[[i]], ...)
  })
  names(result) <- names(object)
  result
}

# make sure an object is a function
resolve_fn <- function(object) {
  if (is.function(object))
    object()
  else
    object
}


# utility function for importing python modules defined in the
# inst/python directory of the package
import_package_module <- function(module, convert = TRUE) {
  python_path <- system.file("python", package = "tfestimators")
  import_from_path(module, python_path, convert = convert)
}

resolve_mode <- function() {
  calls <- sys.calls()
  
  mode <- list(
    infer = ~ identical(.x[[1]], quote(predict.tf_estimator)) ||
      identical(.x[[1]], quote(object$estimator$predict)),
    eval = ~ identical(.x[[1]], quote(evaluate.tf_estimator)) ||
      identical(.x[[1]], quote(object$estimator$evaluate))
  ) %>%
    purrr::map(~ purrr::detect(calls, .x)) %>%
    purrr::compact() %>%
    names()
  
  if (!length(mode))
    stop("no train() or evaluate() function detected in call stack")
  if (length(mode) > 1)
    stop("both train() and evaluate() detected in call stack")
  
  mode
}

compose_history_metadata <- function(history, max_rows = 100, rename_step_col = TRUE) {
  training_history <- as.data.frame(history) %>%
    tidyr::spread("metric", "value")
  
  training_history <- if (nrow(training_history) > max_rows) {
    # cap number of points plotted
    nrow_history <- nrow(training_history)
    sampling_indices <- seq(1, nrow_history, by = nrow_history / max_rows) %>%
      as.integer()
    training_history[sampling_indices,]
  } else training_history
  
  if (rename_step_col)
    names(training_history)[names(training_history) == "step"] <- "epoch"
  training_history
}
