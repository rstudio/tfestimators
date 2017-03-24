context("Testing tf_custom_models methods")

test_that("custom model works on iris data", {

  constructed_input_fn <- input_fn(
    x = iris,
    response = "Species",
    features = c(
      "Sepal.Length",
      "Sepal.Width",
      "Petal.Length",
      "Petal.Width"),
    features_as_named_list = FALSE
  )

  custom_model_fn <- function(features, labels, mode, params, config) {
    labels <- tf$one_hot(labels, 3L)

    # Create three fully connected layers respectively of size 10, 20, and 10 with
    # each layer having a dropout probability of 0.1.
    logits <- features %>%
      tf$contrib$layers$stack(
        tf$contrib$layers$fully_connected, c(10L, 20L, 10L),
        normalizer_fn = tf$contrib$layers$dropout,
        normalizer_params = list(keep_prob = 0.9)) %>%
      tf$contrib$layers$fully_connected(3L, activation_fn = NULL) # Compute logits (1 per class) and compute loss.

    loss <- tf$losses$softmax_cross_entropy(labels, logits)
    predictions <- list(
      class = tf$argmax(logits, 1L),
      prob = tf$nn$softmax(logits))

    # Create a tensor for training op.
    train_op <- tf$contrib$layers$optimize_loss(
      loss,
      tf$contrib$framework$get_global_step(),
      optimizer = 'Adagrad',
      learning_rate = 0.1)

    return(estimator_spec(predictions, loss, train_op, mode))
  }

  classifier <- estimator(
      model_fn = custom_model_fn) %>%
    fit(input_fn = constructed_input_fn)
  ## TODO: Fix the issue on Python end
  # predictions <- predict(classifier, input_fn = constructed_input_fn, type = "prob")
  # expect_equal(length(predictions), 150 * length(unique(iris$Species)))
  # expect_lte(max(predictions), 1)
  # expect_gte(min(predictions), 0)
  # coef s3 method
  # expect_gt(length(coef(classifier)), 1) # TODO: Add get_variable_names attr on Python end
  expect_equal(names(evaluate(classifier, constructed_input_fn)), c("loss", "global_step"))
})

test_that("custom model works on mtcars data", {

  mtcars$vs <- as.factor(mtcars$vs)
  constructed_input_fn <- input_fn(
    mtcars,
    response = "vs",
    features = c("drat", "cyl"),
    features_as_named_list = FALSE
  )

  custom_model_fn <- function(features, labels, mode, params, config) {
    labels <- tf$one_hot(labels, 2L)

    # Create three fully connected layers respectively of size 10, 20, and 10 with
    # each layer having a dropout probability of 0.1.
    logits <- features %>%
      tf$contrib$layers$stack(
        tf$contrib$layers$fully_connected, c(10L, 20L, 10L),
        normalizer_fn = tf$contrib$layers$dropout,
        normalizer_params = list(keep_prob = 0.9)) %>%
      tf$contrib$layers$fully_connected(2L, activation_fn = NULL) # Compute logits (1 per class) and compute loss.

    loss <- tf$losses$softmax_cross_entropy(labels, logits)
    predictions <- list(
      class = tf$argmax(logits, 1L),
      prob = tf$nn$softmax(logits))

    # Create a tensor for training op.
    train_op <- tf$contrib$layers$optimize_loss(
      loss,
      tf$contrib$framework$get_global_step(),
      optimizer = 'Adagrad',
      learning_rate = 0.1)

    return(estimator_spec(predictions, loss, train_op, mode))
  }
  
  classifier <- estimator(
    model_fn = custom_model_fn) %>%
    fit(input_fn = constructed_input_fn)
})

