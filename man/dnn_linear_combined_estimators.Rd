% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dnn_linear_combined_estimators.R
\name{dnn_linear_combined_estimators}
\alias{dnn_linear_combined_estimators}
\alias{dnn_linear_combined_regressor}
\alias{dnn_linear_combined_classifier}
\title{Linear Combined Deep Neural Networks}
\usage{
dnn_linear_combined_regressor(
  model_dir = NULL,
  linear_feature_columns = NULL,
  linear_optimizer = "Ftrl",
  dnn_feature_columns = NULL,
  dnn_optimizer = "Adagrad",
  dnn_hidden_units = NULL,
  dnn_activation_fn = "relu",
  dnn_dropout = NULL,
  label_dimension = 1L,
  weight_column = NULL,
  input_layer_partitioner = NULL,
  config = NULL
)

dnn_linear_combined_classifier(
  model_dir = NULL,
  linear_feature_columns = NULL,
  linear_optimizer = "Ftrl",
  dnn_feature_columns = NULL,
  dnn_optimizer = "Adagrad",
  dnn_hidden_units = NULL,
  dnn_activation_fn = "relu",
  dnn_dropout = NULL,
  n_classes = 2L,
  weight_column = NULL,
  label_vocabulary = NULL,
  input_layer_partitioner = NULL,
  config = NULL
)
}
\arguments{
\item{model_dir}{Directory to save the model parameters, graph, and so on.
This can also be used to load checkpoints from the directory into a
estimator to continue training a previously saved model.}

\item{linear_feature_columns}{The feature columns used by linear (wide) part
of the model.}

\item{linear_optimizer}{Either the name of the optimizer to be used when
training the model, or a TensorFlow optimizer instance. Defaults to the
FTRL optimizer.}

\item{dnn_feature_columns}{The feature columns used by the neural network
(deep) part in the model.}

\item{dnn_optimizer}{Either the name of the optimizer to be used when
training the model, or a TensorFlow optimizer instance. Defaults to the
Adagrad optimizer.}

\item{dnn_hidden_units}{An integer vector, indicating the number of hidden
units in each layer. All layers are fully connected. For example,
\code{c(64, 32)} means the first layer has 64 nodes, and the second layer
has 32 nodes.}

\item{dnn_activation_fn}{The activation function to apply to each layer. This can either be an
actual activation function (e.g. \code{tf$nn$relu}), or the name of an
activation function (e.g. \code{"relu"}). Defaults to the
\code{"relu"} activation function. See
\url{https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn}
for documentation related to the set of activation functions available
in TensorFlow.}

\item{dnn_dropout}{When not \code{NULL}, the probability we will drop out a given
coordinate.}

\item{label_dimension}{Number of regression targets per example. This is the
size of the last dimension of the labels and logits \code{Tensor} objects
(typically, these have shape \verb{[batch_size, label_dimension]}).}

\item{weight_column}{A string, or a numeric column created by
\code{\link[=column_numeric]{column_numeric()}} defining feature column representing weights. It is used
to down weight or boost examples during training. It will be multiplied by
the loss of the example. If it is a string, it is used as a key to fetch
weight tensor from the \code{features} argument. If it is a numeric column,
then the raw tensor is fetched by key \code{weight_column$key}, then
\code{weight_column$normalizer_fn} is applied on it to get weight tensor.}

\item{input_layer_partitioner}{An optional partitioner for the input layer.
Defaults to \code{min_max_variable_partitioner} with \code{min_slice_size} 64 << 20.}

\item{config}{A run configuration created by \code{\link[=run_config]{run_config()}}, used to configure the runtime
settings.}

\item{n_classes}{The number of label classes.}

\item{label_vocabulary}{A list of strings represents possible label values.
If given, labels must be string type and have any value in
\code{label_vocabulary}. If it is not given, that means labels are already
encoded as integer or float within \verb{[0, 1]} for \code{n_classes == 2} and
encoded as integer values in \verb{\{0, 1,..., n_classes  -1\}} for \code{n_classes > 2}. Also there will be errors if vocabulary is not provided and labels are
string.}
}
\description{
Also known as \code{wide-n-deep} estimators, these are estimators for
TensorFlow Linear and DNN joined models for regression.
}
\seealso{
Other canned estimators: 
\code{\link{boosted_trees_estimators}},
\code{\link{dnn_estimators}},
\code{\link{linear_estimators}}
}
\concept{canned estimators}
