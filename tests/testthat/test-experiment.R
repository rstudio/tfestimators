context("Test experiment")

test_that("Experiment works", {
  recipe <- simple_linear_dnn_combined_recipe(mtcars, response = "mpg", linear.features = c("cyl"), dnn.features = c("drat"))
  clf <- linear_dnn_combined_classification(recipe = recipe,
                                            dnn_hidden_units = c(1L, 1L),
                                            dnn_optimizer = "Adagrad",
                                            skip_fit = TRUE)
  
  experiment <- setup_experiment(estimator = clf$estimator,
                                 train_input_fn = clf$recipe$input.fn,
                                 eval_input_fn = clf$recipe$input.fn,
                                 train_steps = 3L,
                                 eval_steps = 3L)
  experiment_result <- experiment$train_and_evaluate()
  expect_gt(length(experiment_result[[1]]), 1)
})
