context("Test experiment")

test_that("Experiment works", {
  recipe <- simple_linear_dnn_combined_recipe(mtcars, response = "mpg", linear.features = c("cyl"), dnn.features = c("drat"))
  clf <- linear_dnn_combined_classification(recipe = recipe,
                                            dnn_hidden_units = c(1L, 1L),
                                            dnn_optimizer = "Adagrad",
                                            skip_fit = TRUE)
  
  experiment <- setup_experiment(tf_model = clf,
                                 train_data = mtcars,
                                 eval_data = mtcars,
                                 train_steps = 3L,
                                 eval_steps = 3L,
                                 continuous_eval_throttle_secs = 60L)
  experiment_result <- experiment$train_and_evaluate()
  expect_gt(length(experiment_result[[1]]), 1)
  
  # Edge cases
  expect_error(experiment <- setup_experiment(tf_model = clf$estimator,
                                              train_data = mtcars,
                                              eval_data = mtcars,
                                              train_steps = 3L,
                                              eval_steps = 3L,
                                              continuous_eval_throttle_secs = 60L))
  expect_error(experiment <- setup_experiment(tf_model = clf,
                                              train_input_fn = clf$recipe$input.fn,
                                              eval_data = mtcars,
                                              train_steps = 3L,
                                              eval_steps = 3L,
                                              continuous_eval_throttle_secs = 60L))
})
