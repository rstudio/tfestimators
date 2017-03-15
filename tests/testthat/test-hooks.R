context("Testing hooks")

source("utils.R")

test_that("Hooks works with linear dnn combined estimators", {
  specs <- mtcars_regression_specs()
  
  old_verbose_level <- tf$logging$get_verbosity()
  tf$logging$set_verbosity(tf$logging$INFO)
  logging_hook <- tf$python$training$basic_session_run_hooks$LoggingTensorHook
  
  output <- reticulate:::py_capture_output(
    linear_dnn_combined_regressor(
      linear_feature_columns = specs$linear_feature_columns,
      dnn_feature_columns = specs$dnn_feature_columns,
      dnn_hidden_units = c(1L, 1L),
      dnn_optimizer = "Adagrad"
    ) %>% fit(
      input_fn = specs$input_fn,
      steps = 10L,
      monitors = logging_hook(
        tensors = list("global_step"),
        every_n_iter = 2L))
  )
})
