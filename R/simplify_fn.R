simple_simplify_predictions_fn <- function(object, predictions) {
  predictions %>%
    map(~ .x %>%
          map(~.x[[1]]) %>%
          flatten() %>% 
          as_tibble()) %>%
    bind_rows()
}

simple_simplify_evaluations_fn <- function(object, evaluations) {
  evaluations %>%
    flatten() %>%
    as_tibble() %>%
    glimpse()
}

simplify_results <- function(object, results, simplify, mode_key) {
  if (is.function(simplify)) {
    simplify(results)
  } else {
    if (isTRUE(simplify)) {
      if (is.tf_custom_estimator(object)) {
        warning("Results are not simplified automatically for custom estimators.\n",
                "Custom estimator needs to write a custom simplify function. \n")
        results
      } else {
        if (mode_key == mode_keys()$PREDICT) {
          simple_simplify_predictions_fn(object, results)
        } else if (mode_key == mode_keys()$EVAL) {
          simple_simplify_evaluations_fn(object, results)
        } else {
         stop("simplify_results has only been implemented for predict() and evaluate().") 
        }
      }
    } else {
      results
    }
  }
}
