context("Testing linear estimators")

test_that("linear_regressor() produces similar fits to lm()", {
  constructed_feature_columns <- construct_feature_columns(mtcars, "drat")
  constructed_input_fn <- construct_input_fn(mtcars, "mpg", "drat")

  tf_model <- linear_regressor(feature_columns = constructed_feature_columns) %>%
    fit(input_fn = constructed_input_fn)
  rs_model <- lm(mpg ~ drat, data = mtcars)

  tf_coef <- coef(tf_model)
  rs_coef <- coef(rs_model)

  predictions <- predict(tf_model, input_fn = constructed_input_fn)
  expect_equal(length(predictions), 32)
})

test_that("linear_classifier() runs successfully", {
  constructed_feature_columns <- construct_feature_columns(mtcars, "drat")
  constructed_input_fn <- construct_input_fn(mtcars, "vs", "drat")

  tf_model <- linear_classifier(feature_columns = constructed_feature_columns) %>%
    fit(input_fn = constructed_input_fn)
  tf_coef <- coef(tf_model)

  predictions <- predict(tf_model, input_fn = constructed_input_fn)
  expect_equal(length(predictions), 32)
})
