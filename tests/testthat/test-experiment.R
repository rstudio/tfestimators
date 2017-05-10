context("Testing experiment")

source("utils.R")

test_that("Experiment works", {
  
  specs <- mtcars_classification_specs()

  clf <-
    linear_dnn_combined_classifier(
      linear_feature_columns = specs$linear_feature_columns,
      dnn_feature_columns = specs$dnn_feature_columns,
      dnn_hidden_units = c(3L, 3L),
      dnn_optimizer = "Adagrad"
    ) %>% fit(input_fn = specs$input_fn)

  experiment <- experiment(
    clf,
    train_input_fn = specs$input_fn,
    eval_input_fn = specs$input_fn,
    train_steps = 3L,
    eval_steps = 3L,
    continuous_eval_throttle_secs = 60L
  )
  
  exp_fn <- function(output_dir) {experiment$experiment}
  learn_runner <- contrib_learn$python$learn$learn_runner
  tmp_dir <- tempfile()
  dir.create(tmp_dir)
  result <- learn_runner$run(
    experiment_fn = exp_fn,
    output_dir = tmp_dir,
    schedule = "local_run"
  )
  expect_gt(length(result[[1]]), 1)
  
  experiment_result <- train_and_evaluate(experiment)
  expect_gt(length(experiment_result[[1]]), 1)
})
