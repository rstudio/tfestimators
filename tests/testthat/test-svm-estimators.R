context("Testing svm estimators")

test_that("svm_classification() runs successfully on fake data", {
  fake_data <- data.frame(
    id_column = c("a", "b", "c"),
    feature1 = c(0.5, 1.0, 1.0),
    feature2 = c(1.0, -1.0, 0.5),
    target = c(1, 0, 1))
  columns <- feature_columns(fake_data, c("feature1", "feature2"))
  constructed_input_fn <- input_fn(
    fake_data,
    response = "target",
    features = c("feature1", "feature2", "id_column"),
    features_as_named_list = T
  )

  # clf <-
  #   svm_classifier(
  #     feature_columns = columns,
  #     example_id_column = "id_column"
  #   ) %>% fit(input_fn = constructed_input_fn)
})

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