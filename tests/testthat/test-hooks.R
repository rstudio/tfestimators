context("Testing hooks")

test_that("Hooks works with linear_regressor", {
  specs <- mtcars_regression_specs()

  lr <- linear_regressor(feature_columns = specs$linear_feature_columns)
  lr %>% train(
      input_fn = specs$input_fn,
      steps = 10,
      hooks = hook_logging_tensor(
        tensors = list("global_step"),
        every_n_iter = 2))

  lr <- linear_regressor(feature_columns = specs$linear_feature_columns)
  lr %>% train(
    input_fn = specs$input_fn,
    steps = 10,
    hooks = hook_checkpoint_saver(
      checkpoint_dir = "/tmp/ckpt_dir",
      save_secs = 2))
  expect_true(length(list.files("/tmp/ckpt_dir")) > 1)
})

test_that("Custom hooks work with linear_regressor", {
  specs <- mtcars_regression_specs()
  custom_hook <- session_run_hook(
    end = function(session) {
      cat(expected_output)
    }
  )
  
  expected_output <- "Running custom session run hook at the end of a session"

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

test_that("Built-in Custom Hook works with linear_regressor", {
  specs <- mtcars_regression_specs()
  
  # Test hook_progress_bar
  lr <- linear_regressor(feature_columns = specs$linear_feature_columns) 
  training_history <- lr %>% train(
    input_fn = specs$input_fn,
    steps = 2,
    hooks = list(
      hook_progress_bar()
    ))
  lr %>% evaluate(
    input_fn = specs$input_fn,
    steps = 2,
    hooks = list(
      hook_progress_bar()))
  # verify that history is not saved by progress bar
  expect_equal(dim(as.data.frame(training_history)), c(0, 0))
  expect_equal(
    lapply(tfestimators:::.globals$history, function(x) dim(as.data.frame(x))),
    list(train = c(0, 0), eval = c(0, 0))
  )
  
  # Test hook_history_saver
  lr <- linear_regressor(feature_columns = specs$linear_feature_columns)
  training_history <- lr %>% train(
    input_fn = specs$input_fn,
    steps = 10,
    hooks = list(
      hook_history_saver(every_n_step = 2)
    ))
  lr %>% evaluate(
    input_fn = specs$input_fn,
    steps = 10,
    hooks = list(
      hook_history_saver(every_n_step = 2)))
  # verify history is saved for both training and evaluation
  expect_equal(
    lapply(tfestimators:::.globals$history, function(x) dim(as.data.frame(x))),
    list(train = 2:3, eval = 2:3)
  )
  expect_equal(dim(as.data.frame(training_history)), c(4, 3))
  
  # Test whether default hooks are attached successfully without any hooks specified
  lr <- linear_regressor(feature_columns = specs$linear_feature_columns)
  training_history <- lr %>% train(
    input_fn = specs$input_fn,
    steps = 2)
  
  # Test whether default hooks are attached successfully with wrapper hooks
  lr <- linear_regressor(feature_columns = specs$linear_feature_columns)
  training_history <- lr %>% train(
    input_fn = specs$input_fn,
    steps = 2,
    hooks = list(
      hook_logging_tensor(
        tensors = list("global_step"),
        every_n_iter = 2),
      hook_checkpoint_saver(
        checkpoint_dir = "/tmp/ckpt_dir",
        save_secs = 2)))
  # Test whether default hooks are attached successfully with wrapper hook and built-in custom hook
  lr <- linear_regressor(feature_columns = specs$linear_feature_columns)
  training_history <- lr %>% train(
    input_fn = specs$input_fn,
    steps = 2,
    hooks = list(
      hook_logging_tensor(
        tensors = list("global_step"),
        every_n_iter = 2),
      hook_history_saver(every_n_step = 2)))
})

