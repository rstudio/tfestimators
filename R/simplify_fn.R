simple_simplify_fn <- function(object, predictions) {
  predictions %>%
    map(~ .x %>%
          map(~.x[[1]]) %>%
          flatten() %>% 
          as_tibble()) %>%
    bind_rows()
}

validate_prediction_simplify_fn <- function(prediction_simplify_fn) {
  if (length(formals(prediction_simplify_fn)) == 1) {
    TRUE
  } else {
    stop("simplify must be a function with only one argument.")
  }
}

simplify_predictions <- function(object, predictions, simplify) {
  if (is.function(simplify) && validate_prediction_simplify_fn(simplify)) {
    simplify(predictions)
  } else {
    if (isTRUE(simplify)) {
      if (is.tf_custom_estimator(object)) {
        warning("Predictions are not simplified automatically for custom estimators.\n",
                "Custom estimator needs to write a custom prediction simplify function. \n",
                "See ?predict.tf_estimator for more details.")
        predictions
      } else {
        # Apply a simple prediciton simplify function for canned estimators
        simple_simplify_fn(object, predictions)
      }
    } else {
      predictions
    }
  }
}
