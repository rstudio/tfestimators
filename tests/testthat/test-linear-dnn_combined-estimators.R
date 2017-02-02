context("Test linear dnn combined estimators")

test_that("linear_dnn_combined_regression() runs successfully", {
  recipe <- simple_linear_dnn_combined_recipe(mtcars, response = "mpg", linear.features = c("cyl"), dnn.features = c("drat"))
  reg <- linear_dnn_combined_regression(recipe = recipe, dnn_hidden_units = c(1L, 1L), dnn_optimizer = "Adagrad")
  coefs <- coef(reg)

  expect_warning(predictions <- predict(reg))
  expect_equal(length(predictions), 32)
})

test_that("linear_dnn_combined_classification() runs successfully", {
  # Without casting "vs" to factor
  recipe <- simple_linear_dnn_combined_recipe(mtcars, response = "vs", linear.features = c("cyl"), dnn.features = c("drat"))
  reg1 <- linear_dnn_combined_classification(recipe = recipe, dnn_hidden_units = c(3L, 3L), dnn_optimizer = "Adagrad")
  coefs1 <- coef(reg1)
  expect_warning(predictions1 <- predict(reg1))
  expect_equal(length(predictions1), 32)

  # Casting "vs" to factor
  mtcars$vs <- as.factor(mtcars$vs)
  recipe <- simple_linear_dnn_combined_recipe(mtcars, response = "vs", linear.features = c("cyl"), dnn.features = c("drat"))
  reg2 <- linear_dnn_combined_classification(recipe = recipe, dnn_hidden_units = c(3L, 3L), dnn_optimizer = "Adagrad")
  coefs2 <- coef(reg2)
  expect_warning(predictions2 <- predict(reg2))

  # Should be the same model
  expect_equal(predictions1, predictions2)
  expect_equal(coefs1, coefs2)
})
