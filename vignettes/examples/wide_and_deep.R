

#' In this example, we'll introduce how to use the TensorFlow Estimators API to
#' jointly train a wide linear model and a deep feed-forward neural network.
#' This approach combines the strengths of memorization and generalization. It's
#' useful for generic large-scale regression and classification problems with
#' sparse input features (e.g., categorical features with a large number of
#' possible feature values). If you're interested in learning more about how
#' Wide & Deep Learning works, please check out the [white
#' paper](http://arxiv.org/abs/1606.07792).
#'
#' ![Wide & Deep](https://www.tensorflow.org/images/wide_n_deep.svg)
#'
#' The figure above shows a comparison of a wide model (logistic regression with
#' sparse features and transformations), a deep model (feed-forward neural
#' network with an embedding layer and several hidden layers), and a Wide & Deep
#' model (joint training of both). At a high level, there are only 3 steps to
#' configure a wide, deep, or Wide & Deep model using the TF Estimators API:
#'
#' - Select features for the wide part: Choose the sparse base columns and
#' crossed columns you want to use. - Select features for the deep part: Choose
#' the continuous columns, the embedding dimension for each categorical column,
#' and the hidden layer sizes. - Put them all together in a Wide & Deep model
#' (linear_dnn_combined_classifier).
#'
#' And that's it! Let's go through a simple example.
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
gender <- column_categorical_with_vocabulary_list(
  "gender", vocabulary_list = c("Female", "Male"))
education <- column_categorical_with_vocabulary_list(
  "education",
  vocabulary_list = c(
    "Bachelors", "HS-grad", "11th", "Masters", "9th",
    "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
    "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
    "Preschool", "12th"))
marital_status <- column_categorical_with_vocabulary_list(
  "marital_status",
  vocabulary_list = c(
    "Married-civ-spouse", "Divorced", "Married-spouse-absent",
    "Never-married", "Separated", "Married-AF-spouse", "Widowed"))
relationship <- column_categorical_with_vocabulary_list(
  "relationship",
  vocabulary_list = c(
    "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
    "Other-relative"))
workclass <- column_categorical_with_vocabulary_list(
  "workclass",
  vocabulary_list = c(
    "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
    "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"))

# To show an example of hashing:
occupation <- column_categorical_with_hash_bucket(
  "occupation", hash_bucket_size = 1000, dtype = tf$int32)
native_country <- column_categorical_with_hash_bucket(
  "native_country", hash_bucket_size = 1000, dtype = tf$int32)

# Continuous base columns.
age <- column_numeric("age")
education_num <- column_numeric("education_num")
capital_gain <- column_numeric("capital_gain")
capital_loss <- column_numeric("capital_loss")
hours_per_week <- column_numeric("hours_per_week")

# Transformations.
age_buckets <- column_bucketized(
  age, boundaries = c(18, 25, 30, 35, 40, 45, 50, 55, 60, 65))

base_columns <- c(gender, native_country, education, occupation, workclass, relationship, age_buckets)

#' ### The Wide Model: Linear Model with Crossed Feature Columns

#' The wide model is a linear model with a wide set of sparse and crossed feature columns:

crossed_columns <- feature_columns(
  native_country, education, occupation, workclass, relationship, age_buckets,
  column_crossed(c("education", "occupation"), hash_bucket_size = 10000),
  column_crossed(c("native_country", "occupation"), hash_bucket_size = 10000),
  column_crossed(c(age_buckets, "education", "occupation"), hash_bucket_size = 10000)
)

#' Wide models with crossed feature columns can memorize sparse interactions
#' between features effectively. That being said, one limitation of crossed
#' feature columns is that they do not generalize to feature combinations that
#' have not appeared in the training data. Let's add a deep model with
#' embeddings to fix that.


#' ### The Deep Model: Neural Network with Embeddings

#' The deep model is a feed-forward neural network, as shown in the previous
#' figure. Each of the sparse, high-dimensional categorical features are first
#' converted into a low-dimensional and dense real-valued vector, often referred
#' to as an embedding vector. These low-dimensional dense embedding vectors are
#' concatenated with the continuous features, and then fed into the hidden
#' layers of a neural network in the forward pass. The embedding values are
#' initialized randomly, and are trained along with all other model parameters
#' to minimize the training loss. If you're interested in learning more about
#' embeddings, check out the TensorFlow tutorial on Vector Representations of
#' Words, or Word Embedding on Wikipedia.
#'
#' We'll configure the embeddings for the categorical columns using
#' embedding_column, and concatenate them with the continuous columns:

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

#' The higher the dimension of the embedding is, the more degrees of freedom the
#' model will have to learn the representations of the features. For simplicity,
#' we set the dimension to 8 for all feature columns here. Empirically, a more
#' informed decision for the number of dimensions is to start with a value on
#' the order of $\log_2{n}$ or $k\sqrt[4]{n}$, where n is the number of unique features in a
#' feature column and k is a small constant (usually smaller than 10).
#'
#' Through dense embeddings, deep models can generalize better and make
#' predictions on feature pairs that were previously unseen in the training
#' data. However, it is difficult to learn effective low-dimensional
#' representations for feature columns when the underlying interaction matrix
#' between two feature columns is sparse and high-rank. In such cases, the
#' interaction between most feature pairs should be zero except a few, but dense
#' embeddings will lead to nonzero predictions for all feature pairs, and thus
#' can over-generalize. On the other hand, linear models with crossed features
#' can memorize these “exception rules” effectively with fewer model parameters.
#' Now, let's see how to jointly train wide and deep models and allow them to
#' complement each other’s strengths and weaknesses.

#' ### Combining Wide and Deep Models into One
#'
#' The wide models and deep models are combined by summing up their final output
#' log odds as the prediction, then feeding the prediction to a logistic loss
#' function. All the graph definition and variable allocations have already been
#' handled for you under the hood, so you simply need to create a
#' dnn_linear_combined_classifier:

model <- dnn_linear_combined_classifier(
  linear_feature_columns = crossed_columns,
  dnn_feature_columns = deep_columns,
  dnn_hidden_units = c(100, 50)
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

