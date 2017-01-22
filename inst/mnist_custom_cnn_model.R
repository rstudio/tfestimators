library(tensorflow)

temp_model_dir <- tempfile()
dir.create(temp_model_dir)

mnist_data <- tf$contrib$learn$datasets$load_dataset("mnist", test_with_fake_data = TRUE)

cnn_model_fn <- function(features, labels, mode) {
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer <- tf$reshape(features, c(-1L, 28L, 28L, 1L))

  # Convolutional Layer 1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 <- tf$layers$conv2d(inputs = input_layer, filters = 32L,
                            kernel_size = c(5L, 5L), padding = "same",
                            activation = tf$nn$relu)

  # Pooling Layer 1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 <- tf$layers$max_pooling2d(inputs = conv1, pool_size = c(2L, 2L), strides = 2L)

  # Convolutional Layer 2
  # Computes 64 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 <- tf$layers$conv2d(inputs = pool1, filters = 64L,
                            kernel_size = c(5L, 5L), padding = "same",
                            activation = tf$nn$relu)

  # Pooling Layer 2
  # First max pooling layer with a 2x2 filter and stride of 2
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
    inputs = dense, rate = 0.4, training = (mode == tf$contrib$learn$ModeKeys()$TRAIN))
  
  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits <- tf$layers$dense(inputs = dropout, units = 10L)
  
  loss <- NA
  train_op <- NA
  
  # Calculate Loss (for both TRAIN and EVAL modes)
  if(mode != tf$contrib$learn$ModeKeys()$INFER) {
    onehot_labels <- tf$one_hot(indices = tf$cast(labels, tf$int32), depth = 10L)
    loss <- tf$losses$softmax_cross_entropy(
      onehot_labels = onehot_labels, logits = logits)
  }
  
  # Configure the Training Op (for TRAIN mode)
  if(mode != tf$contrib$learn$ModeKeys()$TRAIN) {
    train_op = tf$contrib$layers$optimize_loss(
      loss = loss,
      global_step = tf$contrib$framework$get_global_step(),
      learning_rate = 0.001,
      optimizer = "SGD")
  }
  
  # Generate Predictions
  predictions <- list(classes = tf$argmax(input = logits, axis = 1L),
                      probabilities = tf$nn$softmax(logits, name = "softmax_tensor"))
  
  # Return a ModelFnOps object
  return(tf$contrib$learn$estimators$model_fn$ModelFnOps(
    mode = mode,
    predictions = predictions,
    loss = loss,
    train_op = train_op
  ))
}

# Initialize classifier using custom cnn model function
classifier <- tf$contrib$learn$Estimator(model_fn = cnn_model_fn, model_dir = temp_model_dir)

# Fit the classifier
classifier$fit(mnist_data$train$images, mnist_data$train$labels, steps = 100L)

# Generate predictiosn on new data
predictions <- classifier$predict(mnist_data$test$images)
# The predictions are iterators by default in Python API so we call iterate() to collect them
predictions <- iterate(predictions)
accuracy <- sum(predictions == mnist_data$train$labels) / length(predictions)
print(paste0("The accuracy is ", accuracy))
