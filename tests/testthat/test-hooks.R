context("Testing hooks")

source("utils.R")

test_that("Hooks works with linear dnn combined estimators", {
  specs <- mtcars_regression_specs()

  lr <- linear_dnn_combined_regressor(
      linear_feature_columns = specs$linear_feature_columns,
      dnn_feature_columns = specs$dnn_feature_columns,
      dnn_hidden_units = c(1L, 1L),
      dnn_optimizer = "Adagrad"
    ) %>% train(
      input_fn = specs$input_fn,
      steps = 10L,
      monitors = hook_logging_tensor(
        tensors = list("global_step"),
        every_n_iter = 2L))
  
  lr <- linear_dnn_combined_regressor(
    linear_feature_columns = specs$linear_feature_columns,
    dnn_feature_columns = specs$dnn_feature_columns,
    dnn_hidden_units = c(1L, 1L),
    dnn_optimizer = "Adagrad"
  ) %>% train(
    input_fn = specs$input_fn,
    steps = 10L,
    monitors = hook_checkpoint_saver(
      checkpoint_dir = "/tmp/ckpt_dir",
      save_secs = 2))
  expect_true(length(list.files("/tmp/ckpt_dir")) > 1)
})

test_that("Custom hooks work with linear dnn combined estimators", {
  specs <- mtcars_regression_specs()
  
  expected_output <- "Running custom session run hook at the end of a session"
  CustomSessionRunHook <- R6::R6Class(
    "CustomSessionRunHook",
    inherit = EstimatorSessionRunHook,
    public = list(
      end = function(session) {
        cat(expected_output)
      }
    )
  )
  
  custom_hook <- CustomSessionRunHook$new()
  
  actual_output <- capture.output(
    linear_dnn_combined_regressor(
      linear_feature_columns = specs$linear_feature_columns,
      dnn_feature_columns = specs$dnn_feature_columns,
      dnn_hidden_units = c(1L, 1L),
      dnn_optimizer = "Adagrad"
    ) %>% train(
      input_fn = specs$input_fn,
      steps = 10L,
      monitors = custom_hook)
  )
  expect_equal(actual_output, expected_output)
})
