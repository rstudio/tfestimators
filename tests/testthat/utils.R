# Util Functions

mtcars_regression_specs <- function() {
  dnn_feature_columns <- feature_columns(mtcars, "drat")
  linear_feature_columns <- feature_columns(mtcars, "cyl")
  constructed_input_fn <- input_fn(mtcars, response = "mpg", features = c("drat", "cyl"))
  list(dnn_feature_columns = dnn_feature_columns,
       linear_feature_columns = linear_feature_columns,
       input_fn = constructed_input_fn)
}

mtcars_classification_specs <- function() {
  mtcars$vs <- as.factor(mtcars$vs)
  dnn_feature_columns <- feature_columns(mtcars, "drat")
  linear_feature_columns <- feature_columns(mtcars, "cyl")
  constructed_input_fn <- input_fn(mtcars, response = "vs", features = c("drat", "cyl"))
  list(dnn_feature_columns = dnn_feature_columns,
       linear_feature_columns = linear_feature_columns,
       input_fn = constructed_input_fn)
}
