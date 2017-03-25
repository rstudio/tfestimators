context("Testing linear estimators")

source("utils.R")

test_that("linear_regressor() runs successfully", {
  specs <- mtcars_regression_specs()

  tf_model <- linear_regressor(feature_columns = specs$linear_feature_columns) %>%
    fit(input_fn = specs$input_fn)

  tf_coef <- coef(tf_model)

  predictions <- predict(tf_model, input_fn = specs$input_fn)
  expect_equal(length(predictions), 32)
})

test_that("linear_classifier() runs successfully", {
  specs <- mtcars_classification_specs()

  tf_model <- linear_classifier(feature_columns = specs$linear_feature_columns) %>%
    fit(input_fn = specs$input_fn)
  tf_coef <- coef(tf_model)

  predictions <- predict(tf_model, input_fn = specs$input_fn)
  expect_equal(length(predictions), 32)
})
