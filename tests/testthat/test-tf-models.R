context("Testing tf models")

test_that("fit() and predict() works for regressors", {
  
  dnn_feature_columns <- construct_feature_columns(mtcars, "drat")
  linear_feature_columns <- construct_feature_columns(mtcars, "cyl")
  constructed_input_fn <- construct_input_fn(mtcars, response = "mpg", features = c("drat", "cyl"))
  
  reg <-
    linear_dnn_combined_regressor(
      linear_feature_columns = linear_feature_columns,
      dnn_feature_columns = dnn_feature_columns,
      dnn_hidden_units = c(1L, 1L),
      dnn_optimizer = "Adagrad"
    ) %>% fit(input_fn = constructed_input_fn)
  
  coefs <- coef(reg)
  expect_gt(length(coefs), 0)
  
  predictions <- predict(reg, input_fn = constructed_input_fn)
  expect_equal(length(predictions), 32)
})

test_that("fit() and predict() works for classifiers", {
  
  mtcars$vs <- as.factor(mtcars$vs)
  dnn_feature_columns <- construct_feature_columns(mtcars, "drat")
  linear_feature_columns <- construct_feature_columns(mtcars, "cyl")
  constructed_input_fn <- construct_input_fn(mtcars, response = "vs", features = c("drat", "cyl"))
  
  clf <-
    linear_dnn_combined_classifier(
      linear_feature_columns = linear_feature_columns,
      dnn_feature_columns = dnn_feature_columns,
      dnn_hidden_units = c(3L, 3L),
      dnn_optimizer = "Adagrad"
    ) %>% fit(input_fn = constructed_input_fn)
  
  coefs <- coef(clf)
  expect_gt(length(coefs), 0)
  
  predictions <- predict(clf, input_fn = constructed_input_fn)
  expect_equal(length(predictions), 32)
  # probabilities
  predictions <- predict(clf, input_fn = constructed_input_fn, type = "prob")
  expect_equal(length(predictions), 64)
  expect_lte(max(predictions), 1)
  expect_gte(min(predictions), 0)
  # other types that is in PredictionKey
  predictions <- predict(clf, input_fn = constructed_input_fn, type = "logistic")
})
