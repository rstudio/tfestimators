context("Testing model save")

source("helper-utils.R")

check_contents <- function(path) {
  dir_contents <- dir(path, recursive = TRUE)
  
  expect_true(any(grepl("saved_model\\.pb", dir_contents)))
  expect_true(any(grepl("variables\\.data", dir_contents)))
  expect_true(any(grepl("variables\\.index", dir_contents)))
  
  unlink(path, recursive = TRUE)
}

export_test_savedmodel <- function(model) {
  temp_path <- file.path(tempfile(), "testthat-save")
  if (dir.exists(temp_path)) unlink(temp_path, recursive = TRUE)
  
  export_savedmodel(model, temp_path)
  
  check_contents(temp_path)
}

test_succeeds("export_savedmodel() runs successfully for linear_regressor", {
  specs <- mtcars_regression_specs()
  
  model <- linear_regressor(feature_columns = specs$linear_feature_columns)
  model %>% train(input_fn = specs$input_fn, steps = 2)

  export_test_savedmodel(model)
})

test_succeeds("export_savedmodel() runs successfully for dnn_linear_combined_regressor", {
  specs <- mtcars_regression_specs()
  
  model <- dnn_linear_combined_regressor(
    linear_feature_columns = specs$linear_feature_columns,
    dnn_feature_columns = specs$dnn_feature_columns,
    dnn_hidden_units = c(3, 3))
  model %>% train(input_fn = specs$input_fn, steps = 2)
  
  export_test_savedmodel(model)
})

test_succeeds("export_savedmodel() runs successfully for dnn_regressor", {
  specs <- mtcars_regression_specs()

  model <- dnn_regressor(
    hidden_units = c(3,3),
    feature_columns = specs$linear_feature_columns
  )
  model %>% train(input_fn = specs$input_fn, steps = 2)

  export_test_savedmodel(model)
})

test_succeeds("export_savedmodel() runs successfully for linear_classifier", {
  specs <- mtcars_classification_specs()
  
  model <- linear_classifier(feature_columns = specs$linear_feature_columns)
  model %>% train(input_fn = specs$input_fn, steps = 2)
  
  export_test_savedmodel(model)
})

test_succeeds("export_savedmodel() runs successfully for dnn_linear_combined_classifier", {
  specs <- mtcars_classification_specs()
  
  model <- dnn_linear_combined_classifier(
    linear_feature_columns = specs$linear_feature_columns,
    dnn_feature_columns = specs$dnn_feature_columns,
    dnn_hidden_units = c(3, 3))
  model %>% train(input_fn = specs$input_fn, steps = 2)
  
  export_test_savedmodel(model)
})

test_succeeds("export_savedmodel() runs successfully for dnn_classifier", {
  specs <- mtcars_classification_specs()
  
  model <- dnn_classifier(
    hidden_units = c(3,3),
    feature_columns = specs$linear_feature_columns
  )
  model %>% train(input_fn = specs$input_fn, steps = 2)
  
  export_test_savedmodel(model)
})
