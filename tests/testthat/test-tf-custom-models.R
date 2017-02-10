context("Testing tf_custom_models methods")

test_that("predict() works on a custom model", {

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

    loss <- tf$losses$softmax_cross_entropy(target, logits)

    # Create a tensor for training op.
    train_op <- tf$contrib$layers$optimize_loss(
      loss,
      tf$contrib$framework$get_global_step(),
      optimizer = 'Adagrad',
      learning_rate = 0.1)

    # TODO: This is only for classification but not for regression
    return(custom_model_return_fn(logits, loss, train_op))
  }


  iris_input_fn <- function() {
    features <- tf$constant(as.matrix(iris_data$data))
    labels <- tf$constant(iris_data$target)
    return(list(features, labels))
  }

  # Custom interface
  recipe <- custom_model_recipe(model_fn = custom_model_fn, input_fn = iris_input_fn)

  classifier <- create_custom_estimator(recipe)
  predictions <- predict(classifier, input_fn = iris_input_fn, type = "raw")
  expect_equal(length(predictions), 150)
  expect_equal(max(predictions), 2)
  expect_equal(min(predictions), 0)

  predictions <- predict(classifier, input_fn = iris_input_fn, type = "prob")
  expect_equal(length(predictions), 150 * length(unique(iris_data$target)))
  expect_lte(max(predictions), 1)
  expect_gte(min(predictions), 0)

  # Simple non-formula interface
  simple_recipe <- simple_custom_model_recipe(
    x = iris,
    response = "Species",
    features = c(
      "Sepal.Length",
      "Sepal.Width",
      "Petal.Length",
      "Petal.Width"),
    model_fn = custom_model_fn)

  classifier <- create_custom_estimator(simple_recipe)
  predictions <- predict(classifier, newdata = iris, type = "prob")
  expect_equal(length(predictions), 150 * length(unique(iris$Species)))
  expect_lte(max(predictions), 1)
  expect_gte(min(predictions), 0)
  
  # Formula interface
  formula_recipe <- simple_custom_model_recipe(
    Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
    data = iris,
    model_fn = custom_model_fn)
  
  classifier <- create_custom_estimator(simple_recipe)
  predictions <- predict(classifier, newdata = iris, type = "prob")
  expect_equal(length(predictions), 150 * length(unique(iris$Species)))
  expect_lte(max(predictions), 1)
  expect_gte(min(predictions), 0)
})
