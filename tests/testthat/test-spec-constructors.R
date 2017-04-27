context("Testing spec constructors")

test_that("feature columns can be constructed correctly", {
  
  fcs <- feature_columns(mtcars, "drat")
  expect_equal(length(fcs()), 1)
  expect_true(grepl("RealValuedColumn", class(fcs()[[1]])[1]))
  fcs <- feature_columns(mtcars, c("drat", "cyl"))
  expect_equal(length(fcs()), 2)
  expect_true(grepl("RealValuedColumn", class(fcs()[[1]])[1]))
})

test_that("input_fn can be constructed correctly using custom input_fn", {

  features <- c("drat", "cyl")
  input_fn1 <- input_fn(mtcars, response = "mpg", features = features)$input_fn
  expect_equal(length(input_fn1()), 2)
  expect_equal(length(input_fn1()[[1]]), length(features))
  
  # through formula interface
  input_fn2 <- input_fn(mpg ~ drat + cyl, data = mtcars)$input_fn
  expect_equal(length(input_fn2()), 2)
  expect_equal(length(input_fn2()[[1]]), length(features))
  
  expect_equal(input_fn1, input_fn2)
})

test_that("input_fn can be constructed correctly from data.frame objects", {
  
  features <- c("drat", "cyl")
  input_fn1 <- input_fn(mtcars, response = "mpg", features = features)$input_fn
  expect_equal(length(input_fn1()), 2)
  expect_equal(length(input_fn1()[[1]]), length(features))
})

test_that("input_fn can be constructed correctly from list objects", {
  fake_sequence_input_fn <-
    input_fn(
      x = list(
        features = list(
          list(list(1), list(2), list(3)),
          list(list(4), list(5), list(6))),
        response = list(
          list(1, 2, 3), list(4, 5, 6))),
      features = c("features"),
      response = "response")$input_fn()

  expect_equal(length(fake_sequence_input_fn), 2)
  expect_true(inherits(fake_sequence_input_fn[[1]][[1]], "tensorflow.python.framework.ops.Tensor"))
  expect_true(inherits(fake_sequence_input_fn[[2]], "tensorflow.python.framework.ops.Tensor"))
})

