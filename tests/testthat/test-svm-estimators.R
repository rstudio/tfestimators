context("Testing svm estimators")

test_that("svm_classifier() runs successfully", {

  # mtcars$vs <- as.numeric(mtcars$vs) - 1 # To remove

  # mtcars$vs <- as.factor(mtcars$vs)
  # feature_columns <- function() {
  #   construct_feature_columns(mtcars, c("cyl", "drat"))
  # }
  # constructed_input_fn <- construct_input_fn(mtcars,
  #                                            response = "vs",
  #                                            features = c("drat", "cyl"),
  #                                            id_column = "id_column")
  # 
  # clf <-
  #   svm_classifier(
  #     feature_columns = feature_columns,
  #     example_id_column = "id_column",
  #     weight_column_name = NULL
  #   ) %>% fit(input_fn = constructed_input_fn)

  # coefs <- coef(clf)
  # 
  # expect_warning(predictions <- predict(clf))
})
