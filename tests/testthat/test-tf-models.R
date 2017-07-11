context("Testing tf models")

test_that("train(), predict(), and evaluate() work for regressors", {
  
  specs <- mtcars_regression_specs()

  reg <-
    dnn_linear_combined_regressor(
      linear_feature_columns = specs$linear_feature_columns,
      dnn_feature_columns = specs$dnn_feature_columns,
      dnn_hidden_units = c(1L, 1L),
      dnn_optimizer = "Adagrad"
    ) %>% train(input_fn = specs$input_fn, steps = 2)
  
  coefs <- coef(reg)
  expect_gt(length(coefs), 0)
  
  predictions <- predict(reg, input_fn = specs$input_fn)
  expect_equal(length(predictions), 32)
  
  loss <- evaluate(reg, input_fn = specs$input_fn)$loss
  expect_lte(loss, 4000)
})

test_that("train(), predict(), and evaluate() work for classifiers", {
  
  specs <- mtcars_classification_specs()

  tmp_dir <- tempdir()
  dir.create(tmp_dir, showWarnings = FALSE)
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

  predictions <- predict(clf, input_fn = specs$input_fn)
  expect_equal(length(predictions), 32)
  # probabilities
  predictions <- unlist(predict(clf, input_fn = specs$input_fn, predict_keys = prediction_keys()$PROBABILITIES))
  expect_equal(length(predictions), 64)
  expect_lte(max(predictions), 1)
  expect_gte(min(predictions), 0)
  # other types that is in PredictionKey
  predictions <- predict(clf, input_fn = specs$input_fn, predict_keys = prediction_keys()$LOGISTIC)

  accuracy <- evaluate(clf, input_fn = specs$input_fn)$accuracy
  expect_lte(accuracy, 0.6)

  unlink(tmp_dir, recursive = TRUE)
})
