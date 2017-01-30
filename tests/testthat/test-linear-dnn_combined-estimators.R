context("Test linear dnn combined estimators")

test_that("linear_dnn_combined_regression() runs successfully", {
  recipe <- simple_linear_dnn_combined_recipe(mtcars, response = "mpg", linear.features = c("cyl"), dnn.features = c("drat"))
  linear_dnn_combined_regression(recipe = recipe, dnn_hidden_units = c(10L, 10L, 10L), dnn_optimizer = "Adagrad")
})
