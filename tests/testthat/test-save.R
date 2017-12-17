context("Testing model save")

source("helper-utils.R")

check_contents <- function(path) {
  dir_contents <- dir(path, recursive = TRUE)
  
  expect_true(any(grepl("saved_model\\.pb", dir_contents)))
  expect_true(any(grepl("variables\\.data", dir_contents)))
  expect_true(any(grepl("variables\\.index", dir_contents)))
  
  unlink(path, recursive = TRUE)
}

test_succeeds("export_savedmodel() runs successfully", {
  specs <- mtcars_regression_specs()
  
  model <- linear_regressor(feature_columns = specs$linear_feature_columns)
  model %>% train(input_fn = specs$input_fn, steps = 2)
  
  temp_path <- file.path(tempdir(), "testthat-save")
  if (dir.exists(temp_path)) unlink(temp_path, recursive = TRUE)
  
  export_savedmodel(model, temp_path)
  
  check_contents(temp_path)
  
  # Test canned estimators with multiple feature columns in args
  specs <- mtcars_regression_specs()
  model <- dnn_linear_combined_regressor(
    linear_feature_columns = specs$linear_feature_columns,
    dnn_feature_columns = specs$dnn_feature_columns,
    dnn_hidden_units = c(3, 3))
  model %>% train(input_fn = specs$input_fn, steps = 2)
  export_savedmodel(model, temp_path)
  check_contents(temp_path)
})
