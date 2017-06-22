

#' In this example, we'll introduce how to use the TensorFlow Estimators API to
#' jointly train a wide linear model and a deep feed-forward neural network.
#' This approach combines the strengths of memorization and generalization. It's
#' useful for generic large-scale regression and classification problems with
#' sparse input features (e.g., categorical features with a large number of
#' possible feature values). If you're interested in learning more about how
#' Wide & Deep Learning works, please check out the [white
#' paper](http://arxiv.org/abs/1606.07792).
#' 
#'
#' ### Download Data
#' 
#' First of all, let's download the census data:

library(tfestimators)

maybe_download_census <- function(train_data_path, test_data_path, column_names_to_assign) {
  if (!file.exists(train_data_path) || !file.exists(test_data_path)) {
    cat("Downloading census data ...")
    train_data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header = FALSE, skip = 1)
    test_data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", header = FALSE, skip = 1)
    colnames(train_data) <- column_names_to_assign
    colnames(test_data) <- column_names_to_assign
    write.csv(train_data, train_data_path, row.names = FALSE)
    write.csv(test_data, test_data_path, row.names = FALSE)
  } else {
    train_data <- read.csv(train_data_path, header = TRUE)
    test_data <- read.csv(test_data_path, header = TRUE)
  }
  return(list(train_data = train_data, test_data = test_data))
}

COLNAMES <- c("age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
              "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country",
              "income_bracket")

downloaded_data <- maybe_download_census(
  file.path(getwd(), "train_census.csv"),
  file.path(getwd(), "test_census.csv"),
  COLNAMES
)
train_data <- downloaded_data$train_data
test_data <- downloaded_data$test_data

#' ### Define Base Feature Columns
#' 
#' Next, let's define the base categorical and continuous feature columns that
#' we'll use. These base columns will be the building blocks used by both the
#' wide part and the deep part of the model.
#'

# Categorical base columns
education <- column_categorical_with_hash_bucket("education", hash_bucket_size = 1000, dtype = tf$int32)
relationship <- column_categorical_with_hash_bucket("relationship", hash_bucket_size = 100, dtype = tf$int32)
workclass <- column_categorical_with_hash_bucket("workclass", hash_bucket_size = 100, dtype = tf$int32)
occupation <- column_categorical_with_hash_bucket("occupation", hash_bucket_size = 100, dtype = tf$int32)
native_country <- column_categorical_with_hash_bucket("native_country", hash_bucket_size = 1000, dtype = tf$int32)

# Continuous base columns.
age <- column_numeric("age")
age_buckets <- column_bucketized(age, boundaries = c(18, 25, 30, 35, 40, 45, 50, 55, 60, 65))
education_num <- column_numeric("education_num")
capital_gain <- column_numeric("capital_gain")
capital_loss <- column_numeric("capital_loss")
hours_per_week <- column_numeric("hours_per_week")

#' ### Define Wide and Deep Columns

wide_columns <- feature_columns(native_country, education, occupation, workclass, relationship, age_buckets)

deep_columns <- feature_columns(
  column_embedding(workclass, dimension = 8),
  column_embedding(education, dimension = 8),
  column_embedding(relationship, dimension = 8),
  column_embedding(native_country, dimension = 8),
  column_embedding(occupation, dimension = 8),
  age, 
  education_num, 
  capital_gain, 
  capital_loss,
  hours_per_week
)

#' ### Combining Wide and Deep Models into One

model <- linear_dnn_combined_classifier(
  linear_feature_columns = wide_columns,
  dnn_feature_columns = deep_columns,
  dnn_hidden_units = c(100L, 50L)
)

#' ### Training and Evaluating The Model

# Build labels according to income bracket
train_data$income_bracket <- as.character(train_data$income_bracket)
test_data$income_bracket <- as.character(test_data$income_bracket)
train_data$label <- ifelse(train_data$income_bracket == " >50K", 1, 0)
test_data$label <- ifelse(test_data$income_bracket == " >50K", 1, 0)

constructed_input_fn <- function(dataset) {
  input_fn(dataset, features = -label, response = label)
}
train_input_fn <- constructed_input_fn(train_data)
eval_input_fn <- constructed_input_fn(test_data)

train(model, input_fn = train_input_fn, steps = 2)

evaluate(model, input_fn = eval_input_fn, steps = 2)

