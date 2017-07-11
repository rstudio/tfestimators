context("Testing experiment")

test_that("Experiment works", {
  
  specs <- mtcars_regression_specs()

  clf <-
    linear_regressor(
      feature_columns = specs$linear_feature_columns
    ) %>% train(input_fn = specs$input_fn, steps = 2)

  experiment <- experiment(
    clf,
    train_input_fn = specs$input_fn,
    eval_input_fn = specs$input_fn,
    train_steps = 3L,
    eval_steps = 3L,
    continuous_eval_throttle_secs = 60L
  )

  # experiment_result <- evaluate(experiment)
  # expect_gt(length(experiment_result[[1]]), 1)
})
