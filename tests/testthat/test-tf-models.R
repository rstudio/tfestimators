context("Testing tf_models methods")

test_that("predict() can accept new input_fn() from recipe", {
  recipe <- simple_linear_dnn_combined_recipe(mtcars, response = "mpg", linear.features = c("cyl"), dnn.features = c("drat"))
  reg <- linear_dnn_combined_regression(recipe = recipe, dnn_hidden_units = c(1L, 1L), dnn_optimizer = "Adagrad")
  predictions <- predict(reg, input_fn = recipe$input.fn)
  expect_equal(length(predictions), 32)
})

test_that("predict() can produce probabilities of predictions", {
  recipe <- simple_linear_dnn_combined_recipe(mtcars, response = "vs", linear.features = c("cyl"), dnn.features = c("drat"))
  reg <- linear_dnn_combined_classification(recipe = recipe, dnn_hidden_units = c(1L, 1L), dnn_optimizer = "Adagrad")
  prediction_probs <- predict(reg, input_fn = recipe$input.fn, type = "prob")
  expect_true(prediction_probs <= 1 && prediction_probs >=0)
  expect_equal(length(prediction_probs), 64)
})
