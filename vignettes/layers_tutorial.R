devtools::load_all("~/tfestimators/")
library(tensorflow)

cnn_model_fn <- function(features, labels, mode, params, config) {
  
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer <- tf$reshape(features$x, c(-1L, 28L, 28L, 1L))
  
  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 <- tf$layers$conv2d(
    inputs = input_layer,
    filters = 32L,
    kernel_size = c(5L, 5L),
    padding = "same",
    activation = tf$nn$relu)
  
  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 <- tf$layers$max_pooling2d(inputs = conv1, pool_size = c(2L, 2L), strides = 2L)
  
  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 <- tf$layers$conv2d(
    inputs = pool1,
    filters = 64L,
    kernel_size = c(5L, 5L),
    padding = "same",
    activation = tf$nn$relu)
  
  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 <- tf$layers$max_pooling2d(inputs = conv2, pool_size = c(2L, 2L), strides = 2L)
  
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat <- tf$reshape(pool2, c(-1L, 7L * 7L * 64L))
  
  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense <- tf$layers$dense(inputs = pool2_flat, units = 1024L, activation = tf$nn$relu)
  
  # Add dropout operation; 0.6 probability that element will be kept
  dropout <- tf$layers$dropout(
    inputs = dense, rate = 0.4, training = (mode == mode_keys()$TRAIN))
  
  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits <- tf$layers$dense(inputs = dropout, units = 10L)
  
  # Generate Predictions (for PREDICT mode)
  predicted_classes <- tf$argmax(input = logits, axis = 1L)
  if (mode == mode_keys()$PREDICT) {
    predictions <- list(
      classes = predicted_classes,
      probabilities = tf$nn$softmax(logits, name = "softmax_tensor")
    )
    return(estimator_spec(mode = mode, predictions = predictions))
  }
  
  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels <- tf$one_hot(indices = tf$cast(labels, tf$int32), depth = 10L)
  loss <- tf$losses$softmax_cross_entropy(
    onehot_labels = onehot_labels, logits = logits)
  
  # Configure the Training Op (for TRAIN mode)
  if (mode == mode_keys()$TRAIN) {
    optimizer <- tf$train$GradientDescentOptimizer(learning_rate = 0.001)
    train_op <- optimizer$minimize(
      loss = loss,
      global_step = tf$train$get_global_step())
    return(estimator_spec(mode = mode, loss = loss, train_op = train_op))
  }
    
  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops <- list(accuracy = tf$metrics$accuracy(
    labels = labels, predictions = predicted_classes))

  return(estimator_spec(
    mode = mode, loss = loss, eval_metric_ops = eval_metric_ops))
}

np <- import("numpy", convert = FALSE)
# Load training and eval data
mnist <- tf$contrib$learn$datasets$load_dataset("mnist")
train_data <- np$asmatrix(mnist$train$images, dtype = np$float32)
train_labels <- np$asarray(mnist$train$labels, dtype = np$int32)
eval_data <- np$asmatrix(mnist$test$images, dtype = np$float32)
eval_labels <- np$asarray(mnist$test$labels, dtype = np$int32)

# Create the Estimator
mnist_classifier <- estimator(
  model_fn = cnn_model_fn, model_dir = "/tmp/mnist_convnet_model")

# Set up logging for predictions
# Log the values in the "Softmax" tensor with label "probabilities"
# tensors_to_log <- list(probabilities = "softmax_tensor")
# logging_hook <- hook_logging_tensor(
#   tensors = tensors_to_log, every_n_iter=50)

train_input_fn <- function(features_as_named_list) {
  tf$estimator$inputs$numpy_input_fn(
    x = list(x = train_data),
    y = train_labels,
    batch_size = 100L,
    num_epochs = NULL,
    shuffle = TRUE)
}

eval_input_fn <- function(features_as_named_list) {
  tf$estimator$inputs$numpy_input_fn(
    x = list(x = eval_data),
    y = eval_labels,
    batch_size = 100L,
    num_epochs = NULL,
    shuffle = TRUE)
}

train(
  mnist_classifier,
  input_fn = train_input_fn,
  steps = 2)

evaluate(
  mnist_classifier,
  input_fn = eval_input_fn,
  steps = 2)

