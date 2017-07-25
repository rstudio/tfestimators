# Example of DNNClassifier for Iris plant dataset.

library(tfestimators)

response <- function() "Species"
features <- function() setdiff(names(iris), response())

# split into train, test datasets
set.seed(123)
partitions <- modelr::resample_partition(iris, c(test = 0.2, train = 0.8))
iris_train <- as.data.frame(partitions$train)
iris_test  <- as.data.frame(partitions$test)

# construct classifier
classifier <- dnn_classifier(
  feature_columns = feature_columns(
    column_numeric(features())
  ),
  hidden_units = c(10, 20, 10),
  n_classes = 3
)

# construct input function 
.input_fn <- function(data) {
  input_fn(
    data,
    features = features(),
    response = response()
  )
}

# train classifier with training dataset
train(
  classifier,
  input_fn = .input_fn(iris_train)
)

# valuate with test dataset
predictions <- predict(classifier, input_fn = .input_fn(iris_test))
evaluation <- evaluate(classifier, input_fn = .input_fn(iris_test))
