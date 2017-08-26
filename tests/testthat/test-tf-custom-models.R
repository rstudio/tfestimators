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
    
    if (mode == "infer") {
      return(estimator_spec(mode = mode, predictions = predictions, loss = NULL, train_op = NULL))
    }
    
    labels <- tf$one_hot(labels, 3L)
    loss <- tf$losses$softmax_cross_entropy(labels, logits)
    
    # Create a tensor for training op.
    train_op <- tf$contrib$layers$optimize_loss(
      loss,
      tf$contrib$framework$get_global_step(),
      optimizer = 'Adagrad',
      learning_rate = 0.1)
    
    return(estimator_spec(mode = mode, predictions = predictions, loss = loss, train_op = train_op))
  }
  
  tmp_dir <- tempfile()
  
  # training
  classifier <-
    estimator(model_fn = simple_custom_model_fn, model_dir = tmp_dir) %>%
    train(input_fn = constructed_input_fn, steps = 2L)
  
  # check whether tensorboard works with custom estimator
  tensorboard(log_dir = tmp_dir, launch_browser = FALSE)

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

  # validate coefficients
  coefs <- coef(classifier)
  
  # TODO: what test is appropriate here?
  # > str(coefs)
  # List of 12
  # $ OptimizeLoss/Stack/fully_connected_1/weights/Adagrad: num [1:4, 1:10] 0.125 0.105 0.115 0.101 0.1 ...
  # $ OptimizeLoss/Stack/fully_connected_2/weights/Adagrad: num [1:10, 1:20] 0.1 0.1 0.112 0.103 0.1 ...
  # $ OptimizeLoss/Stack/fully_connected_3/weights/Adagrad: num [1:20, 1:10] 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 ...
  # $ OptimizeLoss/fully_connected/biases/Adagrad         : num [1:3(1d)] 0.144 0.34 0.203
  # $ OptimizeLoss/fully_connected/weights/Adagrad        : num [1:10, 1:3] 0.1 0.124 0.103 0.109 0.163 ...
  # $ OptimizeLoss/learning_rate                          : num 0.1
  # $ Stack/fully_connected_1/weights                     : num [1:4, 1:10] -0.108 -0.3476 0.3361 -0.0033 -0.4944 ...
  # $ Stack/fully_connected_2/weights                     : num [1:10, 1:20] 0.044 -0.196 0.201 0.352 -0.274 ...
  # $ Stack/fully_connected_3/weights                     : num [1:20, 1:10] -0.0699 -0.2572 0.1323 0.3049 -0.1709 ...
  # $ fully_connected/biases                              : num [1:3(1d)] 0.0536 -0.0116 -0.0128
  # $ fully_connected/weights                             : num [1:10, 1:3] -0.3485 0.3212 0.1499 -0.0287 -0.0591 ...
  # $ global_step                                         : num 2
})
