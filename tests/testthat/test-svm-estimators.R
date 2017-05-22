context("Testing svm estimators")

test_that("svm_classification() runs successfully on mtcars data", {
  mtcars$vs <- as.factor(mtcars$vs)
  mtcars["id_column"] <- as.character(1:nrow(mtcars))
  columns <- feature_columns(mtcars, c("drat", "cyl"))
  constructed_input_fn <- input_fn(
    mtcars,
    response = "vs",
    features = c("drat", "cyl", "id_column"))
  
  dt <- data.frame(example_id = c("1", "2", "3"),
                   price = c(0.6, 0.8, 0.3),
                   weights = c(3.0, 1.0, 1.0),
                   labels = c(1, 0, 1))
  columns <- feature_columns(dt, c("weights", "price"))
  constructed_input_fn <- input_fn(
    dt,
    response = "labels",
    features = c("price", "weights", "example_id"))

  ## https://github.com/rstudio/tensorflow/issues/104
  ## https://github.com/tensorflow/tensorflow/issues/8043
  clf <-
    svm_classifier(
      feature_columns = columns,
      example_id_column = "example_id",
      weight_column_name = "weights"
    )
  # train(clf, input_fn = constructed_input_fn)
})
