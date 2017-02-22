context("Testing experiment")

test_that("Experiment works", {
  
  recipe <-
    simple_linear_dnn_combined_recipe(
      mtcars,
      response = "mpg",
      linear_features = c("cyl"),
      dnn_features = c("drat")
    )

  clf <- linear_dnn_combined_classification(
    recipe = recipe,
    dnn_hidden_units = c(1L, 1L),
    dnn_optimizer = "Adagrad"
  )

  exp <- setup_experiment(
    clf,
    train_data = mtcars,
    eval_data = mtcars,
    train_steps = 3L,
    eval_steps = 3L,
    continuous_eval_throttle_secs = 60L
  )
  
  exp_fn <- function() {exp$experiment}
  # TODO: Work on learn runner after upgrade to 1.0
  
  experiment_result <- train_and_evaluate(exp)
  expect_gt(length(experiment_result[[1]]), 1)

  exp$experiment


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
    train_input_fn = clf$recipe$input_fn,
    eval_data = mtcars,
    train_steps = 3L,
    eval_steps = 3L,
    continuous_eval_throttle_secs = 60L))
})
