
#' Example 2

library(tfestimators)

maybe_download_abalone <- function(train_data_path, test_data_path, predict_data_path, column_names_to_assign) {
  if (!file.exists(train_data_path) || !file.exists(test_data_path) || !file.exists(predict_data_path)) {
    cat("Downloading abalone data ...")
    train_data <- read.csv("http://download.tensorflow.org/data/abalone_train.csv", header = FALSE)
    test_data <- read.csv("http://download.tensorflow.org/data/abalone_test.csv", header = FALSE)
    predict_data <- read.csv("http://download.tensorflow.org/data/abalone_predict.csv", header = FALSE)
    colnames(train_data) <- column_names_to_assign
    colnames(test_data) <- column_names_to_assign
    colnames(predict_data) <- column_names_to_assign
    write.csv(train_data, train_data_path, row.names = FALSE)
    write.csv(test_data, test_data_path, row.names = FALSE)
    write.csv(predict_data, predict_data_path, row.names = FALSE)
  } else {
    train_data <- read.csv(train_data_path, header = TRUE)
    test_data <- read.csv(test_data_path, header = TRUE)
    predict_data <- read.csv(predict_data_path, header = TRUE)
  }
  return(list(train_data = train_data, test_data = test_data, predict_data = predict_data))
}

COLNAMES <- c("length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "num_rings")

downloaded_data <- maybe_download_abalone(
  file.path(getwd(), "train_abalone.csv"),
  file.path(getwd(), "test_abalone.csv"),
  file.path(getwd(), "predict_abalone.csv"),
  COLNAMES
)
train_data <- downloaded_data$train_data
test_data <- downloaded_data$test_data
predict_data <- downloaded_data$predict_data

custom_model_fn <- function(features, labels, mode, params) {
  
  print(features)
  
  # Connect the first hidden layer to input layer
  first_hidden_layer <- tf$layers$dense(features, 10L, activation = tf$nn$relu)
  
  # Connect the second hidden layer to first hidden layer with relu
  second_hidden_layer <- tf$layers$dense(first_hidden_layer, 10L, activation = tf$nn$relu)
  
  # Connect the output layer to second hidden layer (no activation fn)
  output_layer <- tf$layers$dense(second_hidden_layer, 1L)
  
  # Reshape output layer to 1-dim Tensor to return predictions
  predictions <- tf$reshape(output_layer, c(-1L))
  predictions_list <- list(ages = predictions)
  
  # Calculate loss using mean squared error
  loss <- tf$losses$mean_squared_error(labels, predictions)
  
  eval_metric_ops <- list(
    rmse = tf$metrics$root_mean_squared_error(
      tf$cast(labels, tf$float64), predictions
  ))
  
  optimizer <- tf$train$GradientDescentOptimizer(learning_rate = params$learning_rate)
  train_op <- optimizer$minimize(loss = loss, global_step = tf$train$get_global_step())
  
  return(estimator_spec(
    mode = mode,
    predictions = predictions_list,
    loss = loss,
    train_op = train_op,
    eval_metric_ops = eval_metric_ops
  ))
}


constructed_input_fn <- function(dataset) {
  input_fn(dataset, features = -num_rings, response = num_rings)
}
train_input_fn <- constructed_input_fn(train_data)
test_input_fn <- constructed_input_fn(test_data)
predict_input_fn <- constructed_input_fn(predict_data)


model_params <- list(learning_rate = 0.001)
model <- estimator(custom_model_fn, params = model_params)

train(model, input_fn = train_input_fn, steps = 2)


