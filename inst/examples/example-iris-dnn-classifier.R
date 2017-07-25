# Example of DNNClassifier for Iris plant dataset.

library(tfestimators)

RESPONSE <- "Species"
FEATURES <- setdiff(names(iris), RESPONSE)

# split into train, test datasets
set.seed(123)
n <- nrow(iris)

rows_train <- sort(sample(n, size = 0.75 * n))
rows_test <- setdiff(1:n, rows_train)

iris_train <- iris[rows_train, ]
iris_test  <- iris[rows_test, ]

# construct classifier
classifier <- dnn_classifier(
  feature_columns = feature_columns(
    column_numeric(FEATURES)
  ),
  hidden_units = c(10, 20, 10),
  n_classes = 3
)

# construct input function 
.input_fn <- function(data) {
  input_fn(
    data,
    features = one_of(FEATURES),
    response = one_of(RESPONSE)
  )
}

# train classifier with training dataset
train(
  classifier,
  input_fn = .input_fn(iris_train)
)

# evaluate with test dataset
predictions <- predict(classifier, input_fn = .input_fn(iris_test))
evaluation <- evaluate(classifier, input_fn = .input_fn(iris_test))
