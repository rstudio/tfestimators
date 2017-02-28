context("Testing experiment")

test_that("Experiment works", {
  
  mtcars$vs <- as.factor(mtcars$vs)
  dnn_feature_columns <- function() {
    construct_feature_columns(mtcars, "drat")
  }
  linear_feature_columns <- function() {
    construct_feature_columns(mtcars, "cyl")
  }
  constructed_input_fn <- construct_input_fn(mtcars, response = "vs", features = c("drat", "cyl"))
  
  clf <-
    linear_dnn_combined_classifier(
      linear_feature_columns = linear_feature_columns,
      dnn_feature_columns = dnn_feature_columns,
      dnn_hidden_units = c(3L, 3L),
      dnn_optimizer = "Adagrad"
    ) %>% fit(input_fn = constructed_input_fn)

  exp <- setup_experiment(
    clf,
    train_input_fn = constructed_input_fn,
    eval_input_fn = constructed_input_fn,
    train_steps = 3L,
    eval_steps = 3L,
    continuous_eval_throttle_secs = 60L
  )
  
  exp_fn <- function(output_dir) {exp$experiment}
  learn_runner <- learn$python$learn$learn_runner
  tmp_dir <- tempfile()
  dir.create(tmp_dir)
  result <- learn_runner$run(
    experiment_fn = exp_fn,
    output_dir = tmp_dir,
    schedule = "local_run"
  )
  expect_gt(length(result[[1]]), 1)
  
  experiment_result <- train_and_evaluate(exp)
  expect_gt(length(experiment_result[[1]]), 1)

  # Edge cases
  expect_error(exp <- setup_experiment(
    clf$estimator,
    train_data = mtcars,
    eval_data = mtcars,
    train_steps = 3L,
    eval_steps = 3L,
    continuous_eval_throttle_secs = 60L))

  expect_error(exp <- setup_experiment(
    clf,
    train_input_fn = constructed_input_fn,
    eval_data = mtcars,
    train_steps = 3L,
    eval_steps = 3L,
    continuous_eval_throttle_secs = 60L))
})
