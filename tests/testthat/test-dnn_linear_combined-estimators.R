context("Testing linear dnn combined estimators")

test_that("linear_dnn_combined_regressor() runs successfully", {
  
  specs <- mtcars_regression_specs()
  reg <-
    dnn_linear_combined_regressor(
      linear_feature_columns = specs$linear_feature_columns,
      dnn_feature_columns = specs$dnn_feature_columns,
      dnn_hidden_units = c(1L, 1L),
      dnn_optimizer = "Adagrad"
    ) %>% train(input_fn = specs$input_fn)

  predictions <- predict(reg, input_fn = specs$input_fn)
  expect_equal(length(predictions), 32)
})

test_that("linear_dnn_combined_classifier() runs successfully", {
  
  specs <- mtcars_classification_specs()
  clf <-
    dnn_linear_combined_classifier(
      linear_feature_columns = specs$linear_feature_columns,
      dnn_feature_columns = specs$dnn_feature_columns,
      dnn_hidden_units = c(3L, 3L),
      dnn_optimizer = "Adagrad"
    ) %>% train(input_fn = specs$input_fn)

  predictions <- predict(clf, input_fn = specs$input_fn)
  expect_equal(length(predictions), 32)
})
