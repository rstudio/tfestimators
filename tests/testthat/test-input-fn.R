context("Testing input_fn")

test_that("input_fn can be constructed through formula interface", {

  features <- c("drat", "cyl")
  input_fn1 <- input_fn(mtcars, response = mpg, features = one_of(features))(TRUE)
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
  input_fn1 <- input_fn(mtcars, response = mpg, features = one_of(features))(TRUE)
  expect_equal(length(input_fn1()), 2)
  expect_equal(names(input_fn1()[[1]]), features)
  expect_true(is.tensor(input_fn1()[[1]][[1]]))
  expect_true(is.tensor(input_fn1()[[2]]))
})

test_that("input_fn can be constructed correctly from matrix objects", {
  
  features <- c("drat", "cyl")
  
  # features_as_named_list == TRUE
  input_fn1 <- input_fn(
    as.matrix(mtcars),
    response = mpg,
    features = one_of(features)
  )(TRUE)()
  expect_equal(length(input_fn1), 2)
  expect_equal(names(input_fn1[[1]]), features)
  expect_true(is.tensor(input_fn1[[1]][[1]]))
  expect_true(is.tensor(input_fn1[[2]]))
  
  # features_as_named_list == FALSE
  input_fn1 <- input_fn(
    as.matrix(mtcars),
    response = mpg,
    features = one_of(features)
  )(FALSE)()
  expect_equal(length(input_fn1), 2)
  expect_equal(names(input_fn1), NULL)
  expect_true(is.tensor(input_fn1[[1]][[1]]))
  expect_true(is.tensor(input_fn1[[2]]))
})

test_that("input_fn can be constructed correctly from list objects", {
  fake_sequence_input_fn <-
    input_fn(
      object = list(
        features = list(
          list(list(1), list(2), list(3)),
          list(list(4), list(5), list(6))),
        response = list(
          list(1, 2, 3), list(4, 5, 6))),
      features = c(features),
      response = response)(TRUE)()
  expect_equal(length(fake_sequence_input_fn), 2)
  expect_true(is.tensor(fake_sequence_input_fn[[1]][[1]]))
  expect_true(is.tensor(fake_sequence_input_fn[[2]]))
  
  # features_as_named_list == FALSE
  fake_sequence_input_fn <- input_fn(
     object = list(
       feature1 = list(
         list(list(1), list(2), list(3)),
         list(list(4), list(5), list(6))),
       feature2 = list(
         list(list(7), list(8), list(9)),
         list(list(10), list(11), list(12))),
       response = list(
         list(1, 2, 3), list(4, 5, 6))),
     features = c("feature1", "feature2"),
     response = "response",
     batch_size = 10L)(FALSE)()
  expect_equal(length(fake_sequence_input_fn), 2) # features + response
  expect_true(is.tensor(fake_sequence_input_fn[[1]][[1]]))
  expect_true(is.tensor(fake_sequence_input_fn[[2]]))
  
  # features_as_named_list == TRUE
  fake_sequence_input_fn <- input_fn(
    object = list(
      feature1 = list(
        list(list(1), list(2), list(3)),
        list(list(4), list(5), list(6))),
      feature2 = list(
        list(list(7), list(8), list(9)),
        list(list(10), list(11), list(12))),
      response = list(
        list(1, 2, 3), list(4, 5, 6))),
    features = c("feature1", "feature2"),
    response = "response",
    batch_size = 10L)(TRUE)()
  expect_equal(length(fake_sequence_input_fn), 2) # features + response
  expect_equal(length(fake_sequence_input_fn[[1]]), 2) # two separate features
  expect_true(is.tensor(fake_sequence_input_fn[[1]][[1]])) # first feature
  expect_true(is.tensor(fake_sequence_input_fn[[1]][[2]])) # second feature
  expect_true(is.tensor(fake_sequence_input_fn[[2]]))
})

test_that("R factors are coerced appropriately", {
  
  RESPONSE <- "Species"
  FEATURES <- setdiff(names(iris), RESPONSE)
  
  classifier <- dnn_classifier(
    feature_columns = lapply(FEATURES, column_numeric),
    hidden_units = list(10, 20, 10),
    n_classes = 3
  )
  
  train(
    classifier,
    input_fn = input_fn(
      iris,
      features = one_of(FEATURES),
      response = one_of(RESPONSE)
    )
  )
})
