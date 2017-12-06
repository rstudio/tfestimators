as_tf_prediction <- function(x) {
  structure(x, class = c("tf_prediction", class(x)))
}

#' @export
type_sum.tf_prediction <- function(x) {
  if (is.numeric(x))
    paste0(signif(x, digits = 3), collapse = ", ")
  else
    paste0(x)
}

simple_simplify_predictions_fn <- function(results) {
  results %>%
    purrr::transpose() %>%
    purrr::map(~ purrr::map(.x, as_tf_prediction)) %>% 
    as_tibble()
}

simple_simplify_evaluations_fn <- function(results) {
  results %>%
    rlang::flatten() %>%
    tibble::as_tibble()
}

simplify_results <- function(results, simplify) {
  if (simplify) {
    mode <- resolve_mode()
    switch(mode,
           infer = simple_simplify_predictions_fn,
           eval = simple_simplify_evaluations_fn,
           identity)(results)
  } else results
}
