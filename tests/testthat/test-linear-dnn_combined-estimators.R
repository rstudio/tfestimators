context("Testing linear dnn combined estimators")

test_that("linear_dnn_combined_regressor() runs successfully", {
  
  dnn_feature_columns <- function() {
    construct_feature_columns(mtcars, "drat")
  }
  linear_feature_columns <- function() {
    construct_feature_columns(mtcars, "cyl")
  }
  constructed_input_fn <- construct_input_fn(mtcars, response = "mpg", features = c("drat", "cyl"))
  
  reg <-
    linear_dnn_combined_regressor(
      linear_feature_columns = linear_feature_columns,
      dnn_feature_columns = dnn_feature_columns,
      dnn_hidden_units = c(1L, 1L),
      dnn_optimizer = "Adagrad"
    ) %>% fit(input_fn = constructed_input_fn)
  
  coefs <- coef(reg)

  predictions <- predict(reg, input_fn = constructed_input_fn)
  expect_equal(length(predictions), 32)
})

test_that("linear_dnn_combined_classifier() runs successfully", {
  
  mtcars$vs <- as.factor(mtcars$vs)
  dnn_feature_columns <- function() {
    construct_feature_columns(mtcars, "drat")
  }
  linear_feature_columns <- function() {
    construct_feature_columns(mtcars, "cyl")
  }
  constructed_input_fn <- construct_input_fn(mtcars, response = "vs", features = c("drat", "cyl"))

  clf <-
    linear_dnn_combined_classifier(
      linear_feature_columns = linear_feature_columns,
      dnn_feature_columns = dnn_feature_columns,
      dnn_hidden_units = c(3L, 3L),
      dnn_optimizer = "Adagrad"
    ) %>% fit(input_fn = constructed_input_fn)
  
  coefs <- coef(clf)
  
  predictions <- predict(clf, input_fn = constructed_input_fn)
})
