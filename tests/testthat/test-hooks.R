context("Testing hooks")

source("utils.R")

test_that("Hooks works with linear dnn combined estimators", {
  specs <- mtcars_regression_specs()

  lr <- linear_regressor(
      feature_columns = specs$linear_feature_columns
    ) %>% train(
      input_fn = specs$input_fn,
      steps = 10,
      hooks = hook_logging_tensor(
        tensors = list("global_step"),
        every_n_iter = 2))

  lr <- linear_regressor(
    feature_columns = specs$linear_feature_columns
  ) %>% train(
    input_fn = specs$input_fn,
    steps = 10,
    hooks = hook_checkpoint_saver(
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
    linear_regressor(
      feature_columns = specs$linear_feature_columns
    ) %>% train(
      input_fn = specs$input_fn,
      steps = 10,
      hooks = custom_hook)
  )
  expect_equal(actual_output, expected_output)
})
