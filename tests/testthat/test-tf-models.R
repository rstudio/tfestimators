context("Testing tf models")

test_that("train(), predict(), and evaluate() work for regressors", {
  
  specs <- mtcars_regression_specs()

  estimator <- dnn_linear_combined_regressor(
    linear_feature_columns = specs$linear_feature_columns,
    dnn_feature_columns = specs$dnn_feature_columns,
    dnn_hidden_units = c(1L, 1L),
    dnn_optimizer = "Adagrad"
  )
  
  train(estimator, input_fn = specs$input_fn, steps = 20)
  
  coefs <- coef(estimator)
  expect_gt(length(coefs), 0)
  
  predictions <- predict(estimator, input_fn = specs$input_fn, simplify = FALSE)
  expect_equal(length(predictions), 32)
  
  loss <- evaluate(estimator, input_fn = specs$input_fn)$loss
  expect_lte(loss, 6000)
})

test_that("train(), predict(), and evaluate() work for classifiers", {
  
  specs <- mtcars_classification_specs()

  tmp_dir <- tempfile()
  clf <-
    dnn_linear_combined_classifier(
      linear_feature_columns = specs$linear_feature_columns,
      dnn_feature_columns = specs$dnn_feature_columns,
      dnn_hidden_units = c(3L, 3L),
      dnn_optimizer = "Adagrad",
      model_dir = tmp_dir
    ) %>% train(input_fn = specs$input_fn)

  # check whether tensorboard works with canned estimator
  tensorboard(log_dir = tmp_dir, launch_browser = FALSE)

  coefs <- coef(clf)
  expect_gt(length(coefs), 0)

  predictions <- predict(clf, input_fn = specs$input_fn, simplify = FALSE)
  expect_equal(length(predictions), 32)
  
  # Test prediction simplification for canned estimator
  predictions <- predict(clf, input_fn = specs$input_fn, simplify = TRUE)
  expect_equal(dim(predictions), c(32, 5))
  # Test default of simplify for canned estimator
  predictions <- predict(clf, input_fn = specs$input_fn)
  expect_equal(dim(predictions), c(32, 5))
  
  # probabilities
  predictions <- unlist(predict(clf, input_fn = specs$input_fn, predict_keys = prediction_keys()$PROBABILITIES, simplify = FALSE))
  expect_equal(length(predictions), 64)
  expect_lte(max(predictions), 1)
  expect_gte(min(predictions), 0)
  # other types that is in PredictionKey
  predictions <- predict(clf, input_fn = specs$input_fn, predict_keys = prediction_keys()$LOGISTIC, simplify = FALSE)

  # Evaluation without simplify
  accuracy <- evaluate(clf, input_fn = specs$input_fn, simplify = FALSE)$accuracy
  expect_lte(accuracy, 0.6)
  # Evaluation with simplify
  evaluation_results <- evaluate(clf, input_fn = specs$input_fn, simplify = TRUE)
  expect_equal(dim(evaluation_results), c(1, 9))
})
