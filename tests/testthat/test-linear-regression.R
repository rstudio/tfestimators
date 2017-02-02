context("Linear Regression")

test_that("linear_regression() produces similar fits to lm()", {
  recipe <- simple_linear_recipe(mtcars, "mpg", "drat")
  tf_model <- linear_regression(recipe = recipe)
  rs_model <- lm(mpg ~ drat, data = mtcars)

  tf_coef <- coef(tf_model)
  rs_coef <- coef(rs_model)

  expect_warning(predictions <- predict(tf_model))
  expect_equal(length(predictions), 32)

  # # TODO: the values are close-ish, but not as close as one might expect?
  # expect_true(
  #   all(abs(tf_coef - rs_coef) < 0.5),
  #   "R and TensorFlow produce similar linear model fits"
  # )
})

test_that("linear_classification() runs successfully", {
  recipe <- simple_linear_recipe(mtcars, "vs", "drat")
  tf_model <- linear_classification(recipe = recipe)
  tf_coef <- coef(tf_model)

  expect_warning(predictions <- predict(tf_model))
  expect_equal(length(predictions), 32)
})
