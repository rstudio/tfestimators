library(tflearn)

setup_shortcuts()
boston_dt <- learn_datasets_lib$load_boston()

boston_input_fn <- function() {
  features <- tf$constant(boston_dt$data)
  labels <- tf$constant(boston_dt$target)
  return(list(features, labels))
}

linear_model_fn <- function(features, labels, mode) {

  res <- learn_models_lib$linear_regression(features, labels, init_mean = 0L)
  prediction <- res[[1]]
  loss <- res[[2]]
  train_op <- contrib_optimizers_lib$optimize_loss(loss,
                                                   contrib_variables$get_global_step(),
                                                   optimizer = 'Adagrad',
                                                   learning_rate = 0.1)
  return(list(prediction, loss, train_op))
}

est <- estimators_lib$Estimator(model_fn = linear_model_fn)
est$fit(input_fn = boston_input_fn, steps = 1)
predictions <- iterate(est$predict(input_fn = boston_input_fn))

