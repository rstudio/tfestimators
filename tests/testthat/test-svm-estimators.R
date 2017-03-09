context("Testing svm estimators")

# TODO: Recover this test after https://github.com/tensorflow/tensorflow/pull/8230 is merged
test_that("svm_classification() runs successfully", {

  # mtcars$vs <- as.factor(mtcars$vs)
  # feature_columns <- construct_feature_columns(mtcars, c("drat", "cyl"))
  # constructed_input_fn <- construct_input_fn(mtcars, response = "vs", features = c("drat", "cyl"))
  #
  # clf <-
  #   svm_classifier(
  #     feature_columns = feature_columns,
  #     example_id_column = "id_column",
  #     weight_column_name = "drat"
  #   ) %>% fit()

  # coefs <- coef(clf)
  #
  # expect_warning(predictions <- predict(clf))
})
