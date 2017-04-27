context("Testing svm estimators")

test_that("svm_classification() runs successfully on mtcars data", {
  mtcars$vs <- as.factor(mtcars$vs)
  mtcars["id_column"] <- as.character(1:nrow(mtcars))
  columns <- feature_columns(mtcars, c("drat", "cyl"))
  constructed_input_fn <- input_fn(
    mtcars,
    response = "vs",
    features = c("drat", "cyl", "id_column"))

  ## https://github.com/rstudio/tensorflow/issues/104
  # clf <-
  #   svm_classifier(
  #     feature_columns = columns,
  #     example_id_column = "id_column",
  #     weight_column_name = "drat"
  #   ) %>% fit(input_fn = constructed_input_fn)
})