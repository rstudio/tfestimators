context("Testing tf_custom_models methods")

test_that("custom model works on iris data", {
  
  constructed_input_fn <- input_fn(
    object = iris,
    response = "Species",
    features = c(
      "Sepal.Length",
      "Sepal.Width",
      "Petal.Length",
      "Petal.Width"),
    batch_size = 10L
  )

  tmp_dir <- tempfile()
  
  # training
  classifier <- estimator(model_fn = simple_custom_model_fn, model_dir = tmp_dir) 
  classifier %>% train(input_fn = constructed_input_fn, steps = 2L)
  
  # check whether tensorboard works with custom estimator
  # tensorboard(log_dir = tmp_dir, launch_browser = FALSE)

  # predictions simplified
  predictions <- predict(classifier, input_fn = constructed_input_fn, simplify = TRUE)
  expect_equal(dim(predictions), c(150, 2))
  # predictions not simplified
  predictions <- predict(classifier, input_fn = constructed_input_fn, simplify = FALSE)
  expect_equal(length(predictions), 150)
  
  # extract predicted classes
  predicted_classes <- unlist(lapply(predictions, function(prediction) {
    prediction$class
  }))
  expect_equal(length(predicted_classes), 150)
  
  # extract predicted probabilities
  predicted_probs <- lapply(predictions, function(prediction) {
    prediction$prob
  })
  expect_equal(length(predicted_probs), 150)
  expect_equal(length(unlist(predicted_probs)), 150 * length(unique(iris$Species)))
  expect_lte(max(unlist(predicted_probs)), 1)
  expect_gte(min(unlist(predicted_probs)), 0)
  # each row of probability should sum to 1
  expect_equal(lapply(predictions, function(pred) sum(pred$prob)), rep(list(1), length(predictions)))
  
  # evaluate
  expect_equal(names(evaluate(classifier, constructed_input_fn, steps = 2L, simplify = FALSE)),
               c("loss", "global_step"))
})
