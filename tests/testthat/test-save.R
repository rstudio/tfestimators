context("Testing model save")

test_that("export_savedmodel() runs successfully", {
  specs <- mtcars_regression_specs()
  
  model <- linear_regressor(feature_columns = specs$linear_feature_columns)
  model %>% train(input_fn = specs$input_fn, steps = 2)
  
  temp_path <- file.path(tempdir(), "testthat-save")
  if (dir.exists(temp_path)) unlink(temp_path, recursive = TRUE)
  
  export_savedmodel(model, temp_path)
  
  dir_contents <- dir(temp_path, recursive = TRUE)
  
  expect_true(any(grepl("saved_model\\.pb", dir_contents)))
  expect_true(any(grepl("variables\\.data", dir_contents)))
  expect_true(any(grepl("variables\\.index", dir_contents)))
})
