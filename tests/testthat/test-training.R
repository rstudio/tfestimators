context("Test training methods")

test_that("train_and_evaluate() work for canned estimators", {

  skip_if_tensorflow_below("1.4")

  specs <- mtcars_regression_specs()
  
  est <- dnn_linear_combined_regressor(
    linear_feature_columns = specs$linear_feature_columns,
    dnn_feature_columns = specs$dnn_feature_columns,
    dnn_hidden_units = c(1L, 1L),
    dnn_optimizer = "Adagrad"
  )
  
  tr_spec <- train_spec(input_fn = specs$input_fn, max_steps = 10)
  ev_spec <- eval_spec(input_fn = specs$input_fn, steps = 2)
  train_and_evaluate(
    est,
    train_spec = tr_spec,
    eval_spec = ev_spec
  )
  
})

test_that("train_and_evaluate() work for custom estimators", {
  
  skip_if_tensorflow_below("1.4")

  input <- input_fn(
    object = iris,
    response = "Species",
    features = c(
      "Sepal.Length",
      "Sepal.Width",
      "Petal.Length",
      "Petal.Width"),
    batch_size = 10L
  )
  
  est <- estimator(model_fn = simple_custom_model_fn)
  
  tr_spec <- train_spec(input_fn = input, max_steps = 2)
  ev_spec <- eval_spec(input_fn = input, steps = 2)
  train_and_evaluate(
    est,
    train_spec = tr_spec,
    eval_spec = ev_spec
  )
  
})
