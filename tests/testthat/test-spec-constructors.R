context("Testing tf models")

test_that("feature columns can be constructed correctly", {
  
  feature_columns <- construct_feature_columns(mtcars, "drat")
  expect_equal(length(feature_columns()), 1)
  expect_true(grepl("RealValuedColumn", class(feature_columns()[[1]])[1]))
  feature_columns <- construct_feature_columns(mtcars, c("drat", "cyl"))
  expect_equal(length(feature_columns()), 2)
  expect_true(grepl("RealValuedColumn", class(feature_columns()[[1]])[1]))
})

test_that("input_fn can be constructed correctly", {

  features <- c("drat", "cyl")
  constructed_input_fn <- construct_input_fn(mtcars, response = "mpg", features = features)
  expect_equal(length(constructed_input_fn()), 2)
  expect_equal(length(constructed_input_fn()[[1]]), length(features))
})

test_that("input_fn can be constructed correctly through formula interface", {
  
  features <- c("drat", "cyl")
  constructed_input_fn <- construct_input_fn(mpg ~ drat + cyl, data = mtcars)
  expect_equal(length(constructed_input_fn()), 2)
  expect_equal(length(constructed_input_fn()[[1]]), length(features))
})

