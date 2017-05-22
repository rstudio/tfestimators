context("Testing tf_custom_models methods")

source("utils.R")

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
  
  simple_custom_model_fn <- function(features, labels, mode, params, config) {
    
    # Create three fully connected layers respectively of size 10, 20, and 10 with
    # each layer having a dropout probability of 0.1.
    logits <- features %>%
      tf$contrib$layers$stack(
        tf$contrib$layers$fully_connected, c(10L, 20L, 10L),
        normalizer_fn = tf$contrib$layers$dropout,
        normalizer_params = list(keep_prob = 0.9)) %>%
      tf$contrib$layers$fully_connected(3L, activation_fn = NULL) # Compute logits (1 per class) and compute loss.
    
    predictions <- list(
      class = tf$argmax(logits, 1L),
      prob = tf$nn$softmax(logits))
    
    if(mode == "infer"){
      return(estimator_spec(predictions = predictions, mode = mode, loss = NULL, train_op = NULL))
    }
    
    labels <- tf$one_hot(labels, 3L)
    loss <- tf$losses$softmax_cross_entropy(labels, logits)
    
    # Create a tensor for training op.
    train_op <- tf$contrib$layers$optimize_loss(
      loss,
      tf$contrib$framework$get_global_step(),
      optimizer = 'Adagrad',
      learning_rate = 0.1)
    
    return(estimator_spec(predictions, loss, train_op, mode))
  }
  
  # training
  classifier <- estimator(
    model_fn = simple_custom_model_fn) %>%
    fit(input_fn = constructed_input_fn, steps = 2L)
  
  # inference
  predictions <- predict(classifier, input_fn = constructed_input_fn)
  
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
  expect_equal(names(evaluate(classifier, constructed_input_fn, steps = 2L)), c("loss", "global_step"))
})
