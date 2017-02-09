context("Testing tf_custom_models methods")

test_that("predict() works on a custom model", {
  temp_model_dir <- tempfile()
  dir.create(temp_model_dir)

  iris_data <- learn$datasets$load_dataset("iris")

  custom_model_fn <- function(features, target) {

    target <- tf$one_hot(target, 3L)

    # Create three fully connected layers respectively of size 10, 20, and 10 with
    # each layer having a dropout probability of 0.1.
    logits <- features %>%
      tf$contrib$layers$stack(
        tf$contrib$layers$fully_connected, c(10L, 20L, 10L),
        normalizer_fn = tf$contrib$layers$dropout,
        normalizer_params = list(keep_prob = 0.9)) %>%
      tf$contrib$layers$fully_connected(3L, activation_fn = NULL) # Compute logits (1 per class) and compute loss.

    logits <- tf$contrib$layers$fully_connected(features, 3L, activation_fn = NULL)
    loss <- tf$losses$softmax_cross_entropy(target, logits)

    # Create a tensor for training op.
    train_op <- tf$contrib$layers$optimize_loss(
      loss,
      tf$contrib$framework$get_global_step(),
      optimizer = 'Adagrad',
      learning_rate = 0.1)

    return(custom_model_return_fn(logits, loss, train_op))
  }


  iris_input_fn <- function() {
    features <- tf$constant(as.matrix(iris_data$data))
    labels <- tf$constant(iris_data$target)
    return(list(features, labels))
  }

  config <- learn$estimators$run_config$RunConfig(tf_random_seed=1)

  classifier <- create_custom_estimator(custom_model_fn, iris_input_fn, 2L,
                                        temp_model_dir, config)
  predictions <- predict(classifier, input_fn = iris_input_fn, type = "raw")
  expect_equal(length(predictions), 150)
  expect_equal(max(predictions), 2)
  expect_equal(min(predictions), 0)
  
  predictions <- predict(classifier, input_fn = iris_input_fn, type = "prob")
  expect_equal(length(predictions), 150 * length(unique(iris_data$target)))
  expect_lte(max(predictions), 1)
  expect_gte(min(predictions), 0)
})