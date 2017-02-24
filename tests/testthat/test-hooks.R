context("Testing hooks")


test_that("Hooks works with linear dnn combined estimators", {
  recipe <-
    simple_linear_dnn_combined_recipe(
      mtcars,
      response = "mpg",
      linear_features = c("cyl"),
      dnn_features = c("drat")
    )
  
  old_verbose_level <- tf$logging$get_verbosity()
  tf$logging$set_verbosity(tf$logging$INFO)
  logging_hook <- tf$python$training$basic_session_run_hooks$LoggingTensorHook
  
  output <- reticulate:::py_capture_output(
    linear_dnn_combined_regressor(
      recipe = recipe,
      dnn_hidden_units = c(1L, 1L),
      dnn_optimizer = "Adagrad"
    ) %>% fit(
      steps = 10L,
      monitors = logging_hook(
        tensors = list("global_step"),
        every_n_iter = 2L))
  )
  # tf$logging$set_verbosity(old_verbose_level)
})
