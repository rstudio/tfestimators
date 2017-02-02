context("Test linear dnn combined estimators")

# test_that("linear_dnn_combined_regression() runs successfully", {
#   recipe <- simple_linear_dnn_combined_recipe(mtcars, response = "mpg", linear.features = c("cyl"), dnn.features = c("drat"))
#   reg <- linear_dnn_combined_regression(recipe = recipe, dnn_hidden_units = c(10L, 10L, 10L), dnn_optimizer = "Adagrad")
# 
#   coefs <- coef(reg)
# 
#   predictions <- predict(reg)
#   expect_equal(length(predictions), 32)
# })
# 
# test_that("linear_dnn_combined_classification() runs successfully", {
#   # Without casting "vs" to factor
#   recipe <- simple_linear_dnn_combined_recipe(mtcars, response = "vs", linear.features = c("cyl"), dnn.features = c("drat"))
#   reg1 <- linear_dnn_combined_classification(recipe = recipe, dnn_hidden_units = c(10L, 10L, 10L), dnn_optimizer = "Adagrad")
#   coefs1 <- coef(reg)
#   predictions1 <- predict(reg)
#   expect_equal(length(predictions), 32)
# 
#   # Casting "vs" to factor
#   mtcars$vs <- as.factor(mtcars$vs)
#   recipe <- simple_linear_dnn_combined_recipe(mtcars, response = "vs", linear.features = c("cyl"), dnn.features = c("drat"))
#   reg2 <- linear_dnn_combined_classification(recipe = recipe, dnn_hidden_units = c(10L, 10L, 10L), dnn_optimizer = "Adagrad")
#   coefs2 <- coef(reg)
#   predictions2 <- predict(reg)
#   
#   # Should be the same model
#   expect_equal(predictions1, predictions2)
#   expect_equal(coefs1, coefs2)
# })

test_that("predict() can accept new input_fn() from recipe", {
  recipe <- simple_linear_dnn_combined_recipe(mtcars, response = "mpg", linear.features = c("cyl"), dnn.features = c("drat"))
  reg <- linear_dnn_combined_regression(recipe = recipe, dnn_hidden_units = c(10L, 10L, 10L), dnn_optimizer = "Adagrad")
  predictions <- predict(reg, input_fn = recipe$input.fn)
  expect_equal(length(predictions), 32)
})

test_that("predict() can produce probabilities of predictions", {
  recipe <- simple_linear_dnn_combined_recipe(mtcars, response = "vs", linear.features = c("cyl"), dnn.features = c("drat"))
  reg <- linear_dnn_combined_classification(recipe = recipe, dnn_hidden_units = c(10L, 10L, 10L), dnn_optimizer = "Adagrad")
  prediction_probs <- predict(reg, input_fn = recipe$input.fn, type = "prob")
  expect_true(prediction_probs <= 1 && prediction_probs >=0)
  expect_equal(length(prediction_probs), 64)
})

