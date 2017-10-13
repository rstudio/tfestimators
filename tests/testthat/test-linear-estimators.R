context("Testing linear estimators")

test_that("linear_regressor() runs successfully", {
  specs <- mtcars_regression_specs()

  estimator <-
    linear_regressor(feature_columns = specs$linear_feature_columns) %>%
    train(input_fn = specs$input_fn, steps = 2)

  coef <- coef(estimator)

  predictions <- predict(estimator, input_fn = specs$input_fn, simplify = FALSE)
  expect_equal(length(predictions), 32)
})


test_that("linear_classifier() runs successfully", {
  specs <- mtcars_classification_specs()

  estimator <-
    linear_classifier(feature_columns = specs$linear_feature_columns) %>%
    train(input_fn = specs$input_fn, steps = 2)
  tf_coef <- coef(estimator)

  predictions <- predict(estimator, input_fn = specs$input_fn, simplify = FALSE)
  expect_equal(length(predictions), 32)
})
