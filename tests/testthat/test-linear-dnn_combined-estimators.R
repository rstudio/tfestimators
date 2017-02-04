context("Test linear dnn combined estimators")

test_that("linear_dnn_combined_regression() runs successfully", {
  recipe <- simple_linear_dnn_combined_recipe(mtcars, response = "mpg", linear.features = c("cyl"), dnn.features = c("drat"))
  reg <- linear_dnn_combined_regression(recipe = recipe, dnn_hidden_units = c(1L, 1L), dnn_optimizer = "Adagrad")
  coefs <- coef(reg)

  expect_warning(predictions <- predict(reg))
  expect_equal(length(predictions), 32)
})

test_that("linear_dnn_combined_classification() runs successfully", {

  mtcars$vs <- as.factor(mtcars$vs)
  recipe <- simple_linear_dnn_combined_recipe(mtcars, response = "vs", linear.features = c("cyl"), dnn.features = c("drat"))
  clf <- linear_dnn_combined_classification(recipe = recipe, dnn_hidden_units = c(3L, 3L), dnn_optimizer = "Adagrad")
  coefs <- coef(clf)
  expect_warning(predictions <- predict(clf))
})
