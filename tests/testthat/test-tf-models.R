context("Testing tf_models methods")

test_that("predict() can accept new input_fn() and newdata or use the existing input_fn()", {
  
  recipe <-
    simple_linear_dnn_combined_recipe(
      mtcars,
      response = "mpg",
      linear_features = c("cyl"),
      dnn_features = c("drat")
    )
  
  reg <-
    linear_dnn_combined_regression(
      recipe = recipe,
      dnn_hidden_units = c(1L, 1L),
      dnn_optimizer = "Adagrad"
    ) %>% fit()

  predictions1 <- predict(reg, input_fn = recipe$input_fn)
  expect_equal(length(predictions1), 32)

  predictions2 <- predict(reg, newdata = mtcars)
  expect_equal(predictions1, predictions2)

  expect_warning(predictions3 <- predict(reg))
  expect_equal(length(predictions3), 32)
})

test_that("predict() produces probabilities of predictions for classification problems", {
  
  recipe <-
    simple_linear_dnn_combined_recipe(
      mtcars,
      response = "vs",
      linear_features = c("cyl"),
      dnn_features = c("drat")
    )
  
  clf <-
    linear_dnn_combined_classification(
      recipe = recipe,
      dnn_hidden_units = c(1L, 1L),
      dnn_optimizer = "Adagrad"
    ) %>% fit()
  
  prediction_probs <- predict(clf, input_fn = recipe$input_fn, type = "prob")
  expect_true(prediction_probs <= 1 && prediction_probs >= 0)
  expect_equal(length(prediction_probs), 64)

  recipe <-
    simple_linear_dnn_combined_recipe(
      mtcars,
      response = "mpg",
      linear_features = c("cyl"),
      dnn_features = c("drat")
    )
  
  reg <-
    linear_dnn_combined_regression(
      recipe = recipe,
      dnn_hidden_units = c(1L, 1L),
      dnn_optimizer = "Adagrad"
    ) %>% fit()
  
  expect_error(predict(reg, input_fn = recipe$input_fn, type = "prob"))
})
