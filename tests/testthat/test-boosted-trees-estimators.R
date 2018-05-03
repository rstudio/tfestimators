context("Testing boosted trees estimators")

source("helper-utils.R")

test_succeeds("boosted_trees_regressor() runs successfully", {
  specs <- mtcars_regression_specs()
  
  estimator <- boosted_trees_regressor(
    feature_columns = specs$bucketized_columns,
    n_batches_per_layer = 1)
  estimator %>% train(input_fn = specs$input_fn)
  
  predictions <- predict(estimator, input_fn = specs$input_fn, simplify = FALSE)
  expect_equal(length(predictions), 32)
})


test_succeeds("boosted_trees_classifier() runs successfully", {
  specs <- mtcars_classification_specs()
  
  estimator <- boosted_trees_classifier(
    feature_columns = specs$bucketized_columns,
    n_batches_per_layer = 1)
  estimator %>% train(input_fn = specs$input_fn)
  
  predictions <- predict(estimator, input_fn = specs$input_fn, simplify = FALSE)
  expect_equal(length(predictions), 32)
})
