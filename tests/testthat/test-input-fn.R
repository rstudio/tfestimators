context("Testing input_fn")

test_that("input_fn can be constructed through formula interface", {

  features <- c("drat", "cyl")
  input_fn1 <- input_fn(mtcars, response = "mpg", features = features)(TRUE)
  expect_equal(length(input_fn1()), 2)
  expect_equal(length(input_fn1()[[1]]), length(features))
  
  # through formula interface
  input_fn2 <- input_fn(mpg ~ drat + cyl, data = mtcars)(TRUE)
  expect_equal(length(input_fn2()), 2)
  expect_equal(length(input_fn2()[[1]]), length(features))
  
  expect_equal(input_fn1, input_fn2)
})

test_that("input_fn can be constructed correctly from data.frame objects", {
  
  features <- c("drat", "cyl")
  input_fn1 <- input_fn(mtcars, response = "mpg", features = features)(TRUE)
  expect_equal(length(input_fn1()), 2)
  expect_equal(names(input_fn1()[[1]]), features)
  expect_true(is.tensor(input_fn1()[[1]][[1]]))
  expect_true(is.tensor(input_fn1()[[2]]))
})

test_that("input_fn can be constructed correctly from matrix objects", {
  
  features <- c("drat", "cyl")
  input_fn1 <- input_fn(as.matrix(mtcars), response = "mpg", features = features)(TRUE)
  expect_equal(length(input_fn1()), 2)
  expect_equal(names(input_fn1()[[1]]), features)
  expect_true(is.tensor(input_fn1()[[1]][[1]]))
  expect_true(is.tensor(input_fn1()[[2]]))
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
      response = "response")(TRUE)

  expect_equal(length(fake_sequence_input_fn), 2)
  expect_true(is.tensor(fake_sequence_input_fn[[1]][[1]]))
  expect_true(is.tensor(fake_sequence_input_fn[[2]]))
})
