context("Testing linear estimators")

source("helper-utils.R")

test_succeeds("linear_regressor() runs successfully", {
  specs <- mtcars_regression_specs()

  estimator <- linear_regressor(feature_columns = specs$linear_feature_columns)
  estimator %>% train(input_fn = specs$input_fn, steps = 2)

  predictions <- predict(estimator, input_fn = specs$input_fn, simplify = FALSE)
  expect_equal(length(predictions), 32)
})


test_succeeds("linear_classifier() runs successfully", {
  specs <- mtcars_classification_specs()

  estimator <- linear_classifier(feature_columns = specs$linear_feature_columns)
  estimator %>% train(input_fn = specs$input_fn, steps = 2)

  predictions <- predict(estimator, input_fn = specs$input_fn, simplify = FALSE)
  expect_equal(length(predictions), 32)
})
