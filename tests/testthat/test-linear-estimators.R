context("Testing linear estimators")

source("utils.R")

test_that("linear_regressor() produces similar fits to lm()", {
  specs <- mtcars_regression_specs()

  tf_model <- linear_regressor(feature_columns = specs$linear_feature_columns) %>%
    fit(input_fn = specs$input_fn)
  rs_model <- lm(mpg ~ drat, data = mtcars)

  tf_coef <- coef(tf_model)
  rs_coef <- coef(rs_model)

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
