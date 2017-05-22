context("Testing mutability of estimators")

source("utils.R")

test_that("linear_regressor() is mutable and results are correct", {
  specs <- mtcars_regression_specs()
  
  tf_model <- linear_regressor(feature_columns = specs$linear_feature_columns)
  train(tf_model, input_fn = specs$input_fn)

  tf_model2 <- linear_regressor(feature_columns = specs$linear_feature_columns) %>%
    train(input_fn = specs$input_fn)
  
  tf_coef <- coef(tf_model)
  tf_coef2 <- coef(tf_model2)
  expect_equal(tf_coef, tf_coef2)
  
  predictions <- predict(tf_model, input_fn = specs$input_fn)
  predictions2 <- predict(tf_model2, input_fn = specs$input_fn)
  expect_equal(predictions, predictions2)
  expect_equal(length(predictions), 32)
})
