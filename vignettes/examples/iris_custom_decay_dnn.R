library(tfestimators)

# Construct the input
inputs <- input_fn(
  iris,
  response = "Species",
  features = c(
    "Sepal.Length",
    "Sepal.Width",
    "Petal.Length",
    "Petal.Width"),
  batch_size = 10
)

custom_model_fn <- function(features, labels, mode, params, config) {

  # Create three fully connected layers respectively of size 10, 20, and 10 with
  # each layer having a dropout probability of 0.1.
  logits <- features %>%
    tf$contrib$layers$stack(
    tf$contrib$layers$fully_connected, c(10L, 20L, 10L),
      normalizer_fn = tf$contrib$layers$dropout,
      normalizer_params = list(keep_prob = 0.9)) %>%
    tf$contrib$layers$fully_connected(3L, activation_fn = NULL) # Compute logits (1 per class) and compute loss.

  # Compute predictions.
  predicted_classes <- tf$argmax(logits, 1L)
  if (mode == mode_keys()$PREDICT) {
    predictions <- list(
      class = predicted_classes,
      prob = tf$nn$softmax(logits))
    return(estimator_spec(mode = mode, predictions = predictions))
  }

  # Convert the labels to a one-hot tensor of shape (length of features, 3) and
  # with a on-value of 1 for each one-hot vector of length 3.
  onehot_labels <- tf$one_hot(labels, 3L, 1L, 0L)
  # Compute loss.
  loss <- tf$losses$softmax_cross_entropy(onehot_labels, logits)

  if (mode == mode_keys()$TRAIN) {
    global_step <- tf$train$get_global_step()
    learning_rate <- tf$train$exponential_decay(
      learning_rate = 0.1,
      global_step = global_step,
      decay_steps = 100L,
      decay_rate = 0.001)
    optimizer <- tf$train$AdagradOptimizer(learning_rate = learning_rate)
    train_op <- optimizer$minimize(loss, global_step = global_step)
    return(estimator_spec(mode = mode, loss = loss, train_op = train_op))
  }

  # Compute evaluation metrics.
  eval_metric_ops <- list(
    accuracy = tf$metrics$accuracy(
      labels = labels, predictions = predicted_classes
    ))

  return(estimator_spec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops))
}

model_dir <- "/tmp/iris-custom-decay-cnn-model"

# Intialize classifer and training
classifier <- estimator(
  model_fn = custom_model_fn, model_dir = model_dir)

classifier %>% train(input_fn = inputs, steps = 100)

# Genearate predictions
predictions <- predict(classifier, input_fn = inputs)

# Open TensorBoard
# tensorboard(log_dir = model_dir, launch_browser = FALSE)


