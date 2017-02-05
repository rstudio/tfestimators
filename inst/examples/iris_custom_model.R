library(tflearn)

setup_shortcuts()

temp_model_dir <- tempfile()
dir.create(temp_model_dir)

iris_data <- learn$datasets$load_dataset("iris")

feature_names <- c("V1", "V2", "V3", "V4")
iris_features <- as.data.frame(iris_data$data)
colnames(iris_features) <- feature_names
iris_labels <- iris_data$target

# tf$python$framework$ops$convert_to_tensor(logits)
# tf$python$framework$ops$convert_to_tensor(list(raw_features$V1, raw_features$V2))
# tf$python$framework$ops$convert_to_tensor(list(iris_features$V1, iris_features$V2))

custom_model_fn <- function(features, target) {

  target <- tf$one_hot(target, 3L)

  # Create three fully connected layers respectively of size 10, 20, and 10 with
  # each layer having a dropout probability of 0.1.
  features = tf$contrib$layers$stack(
    features,
    tf$contrib$layers$fully_connected, c(10L, 20L, 10L),
    normalizer_fn = tf$contrib$layers$dropout,
    normalizer_params = list(keep_prob = 0.9))

  # Compute logits (1 per class) and compute loss.
  logits = tf$contrib$layers$fully_connected(features, 3L, activation_fn = NULL)
  loss = tf$losses$softmax_cross_entropy(target, logits)

  # Create a tensor for training op.
  train_op = tf$contrib$layers$optimize_loss(
    loss,
    tf$contrib$framework$get_global_step(),
    optimizer = 'Adagrad',
    learning_rate = 0.1)

  return(c(list(
    class = tf$argmax(logits, 1L),
    prob = tf$nn$softmax(logits)
  ), loss, train_op))
}

iris_input_fn <- function() {
  features <- tf$constant(as.matrix(iris_features))
  labels <- tf$constant(iris_labels)
  return(list(features, labels))
}

config <- learn$estimators$run_config$RunConfig(tf_random_seed=1)

classifier <- tf$contrib$learn$Estimator(
  model_fn = custom_model_fn,
  model_dir = temp_model_dir,
  config = config)

classifier$fit(input_fn = iris_input_fn, steps = 2)

# predictions <- classifier$predict(input_fn = iris_input_fn)
# predictions <- iterate(predictions)
