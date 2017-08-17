#' In this article, we'll develop a custom estimator to be used with the
#' [Abalone dataset](https://archive.ics.uci.edu/ml/datasets/abalone). This
#' dataset provides information on the physical characteristics of a number of
#' abalones (a type of sea snail), and use these characteristics to predict the
#' number of rings in the shell. As described at
#' https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names:
#' 
#' > Predicting the age of abalone from physical measurements.  The age of
#' > abalone is determined by cutting the shell through the cone, staining it,
#' > and counting the number of rings through a microscope -- a boring and
#' > time-consuming task.  Other measurements, which are easier to obtain, are
#' > used to predict the age.  Further information, such as weather patterns
#' > and location (hence food availability) may be required to solve the problem.

library(tfestimators)

#' We'll start by defining a function that will download and save the various
#' abalone datasets we'll use here. These datasets are hosted freely on the
#' TensorFlow website.
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

#' Because the raw datasets are not supplied with column names, we define them
#' explicitly here (in the order they appear in the dataset), and apply them
#' when the datasets are downloaded.
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

#' We now have the abalone datasets available locally. Now, we begin by defining
#' an input function for our soon-to-be-defined estimator. Here, we define an
#' **input function generator** -- this function accepts a dataset, and returns
#' an input function that pulls data from the associated dataset. Using this, we
#' can generate input functions for each of our datasets easily.
#' 
#' Note that we are attempting to predict the `num_rings` variable, and accept
#' all other variables contained within the dataset as potentially associated
#' features.
abalone_input_fn <- function(dataset) {
  input_fn(dataset, features = -num_rings, response = num_rings)
}

#' Next, we define our custom model function. Canned estimators provided by
#' TensorFlow / the `tfestimators` package come with pre-packaged model
#' functions; when you wish to define your own custom estimator, you must
#' provide your own model function. This is the function responsible for
#' constructing the actual neural network to be used in your model, and should
#' be created by composing TensorFlow's primitives for layers together.
#' 
#' We'll construct a network with two fully-connected hidden layers, and
#' a final output layer. After you've constructed your neywork and defined
#' the optimizer + loss functions you wish to use, you can call the
#' `estimator_spec()` function to construct your estimator.
#' 
#' The model function should accept the following parameters:
#' 
#' -  `features`: The feature columns (normally supplied by an input function);
#' 
#' -  `labels`: The true labels, to be used for computing the loss;
#' 
#' -  `mode`: A key that specifies whether training, evaluation, or prediction
#'    is being performs.
#'    
#' -  `params`: A set of custom parameters; typically supplied by the user of
#'     your custom estimator when instances of this estimator are created. (For
#'     example, we'll see later that the `learning_rate` is supplied through
#'     here.)
#' 
#' -  `config`: Runtime configuration values; typically unneeded by custom
#'    estimators, but can be useful if you need to introspect the state of
#'    the associated TensorFlow session.
#'
model_fn <- function(features, labels, mode, params, config) {
  
  # Connect the first hidden layer to input layer
  first_hidden_layer <- tf$layers$dense(features, 10L, activation = tf$nn$relu)
  
  # Connect the second hidden layer to first hidden layer with relu
  second_hidden_layer <- tf$layers$dense(first_hidden_layer, 10L, activation = tf$nn$relu)
  
  # Connect the output layer to second hidden layer (no activation fn)
  output_layer <- tf$layers$dense(second_hidden_layer, 1L)
  
  # Reshape output layer to 1-dim Tensor to return predictions
  predictions <- tf$reshape(output_layer, list(-1L))
  predictions_list <- list(ages = predictions)
  
  # Calculate loss using mean squared error
  loss <- tf$losses$mean_squared_error(labels, predictions)
  
  eval_metric_ops <- list(rmse = tf$metrics$root_mean_squared_error(
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

#' We've defined our model function -- we can now use the `estimator()`
#' function to create an instance of the estimator we've defined, using
#' that model function.
model <- estimator(model_fn, params = list(learning_rate = 0.001))

#' Now, we can train, evaluate, and predict using our estimator.
train(model, input_fn = abalone_input_fn(train_data))
evaluate(model, input_fn = abalone_input_fn(test_data))
predict(model, input_fn = abalone_input_fn(predict_data))
