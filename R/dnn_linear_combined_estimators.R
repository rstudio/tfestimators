#' Linear Combined Deep Neural Networks
#' 
#' Also known as \code{wide-n-deep} estimators, these are estimators for
#' TensorFlow Linear and DNN joined models for regression.
#' 
#' @inheritParams estimators
#' 
#' @template roxlate-activation-fn
#' @templateVar name dnn_activation_fn
#' @templateVar default relu
#' 
#' @param linear_feature_columns The feature columns used by linear (wide) part
#'   of the model.
#' @param linear_optimizer Either the name of the optimizer to be used when
#'   training the model, or a TensorFlow optimizer instance. Defaults to the
#'   FTRL optimizer.
#' @param dnn_feature_columns The feature columns used by the neural network
#'   (deep) part in the model.
#' @param dnn_optimizer Either the name of the optimizer to be used when
#'   training the model, or a TensorFlow optimizer instance. Defaults to the
#'   Adagrad optimizer.
#' @param dnn_hidden_units An integer vector, indicating the number of hidden
#'   units in each layer. All layers are fully connected. For example,
#'   `c(64, 32)` means the first layer has 64 nodes, and the second layer
#'   has 32 nodes.
#' @param dnn_dropout When not NULL, the probability we will drop out a given 
#'   coordinate.
#'   
#' @family canned estimators
#' @name dnn_linear_combined_estimators
NULL

#' @inheritParams dnn_linear_combined_estimators
#' @name dnn_linear_combined_estimators
#' @export
dnn_linear_combined_regressor <- function(model_dir = NULL,
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
                                          config = NULL)
{
  estimator <- py_suppress_warnings(
    tf$estimator$DNNLinearCombinedRegressor(
      model_dir = resolve_model_dir(model_dir),
      linear_feature_columns = ensure_nullable_list(linear_feature_columns),
      linear_optimizer = linear_optimizer,
      dnn_feature_columns = ensure_nullable_list(dnn_feature_columns),
      dnn_optimizer = dnn_optimizer,
      dnn_hidden_units = as.integer(dnn_hidden_units),
      dnn_activation_fn = resolve_activation_fn(dnn_activation_fn),
      dnn_dropout = dnn_dropout,
      label_dimension = as.integer(label_dimension),
      weight_column = weight_column,
      input_layer_partitioner = input_layer_partitioner,
      config = config
    )
  )

  tf_regressor(estimator, "dnn_linear_combined_regressor")
}

#' @inheritParams dnn_linear_combined_estimators
#' @name dnn_linear_combined_estimators
#' @export
dnn_linear_combined_classifier <- function(model_dir = NULL,
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
                                           config = NULL)
{
  estimator <- py_suppress_warnings(
    tf$estimator$DNNLinearCombinedClassifier(
      model_dir = resolve_model_dir(model_dir),
      linear_feature_columns = ensure_nullable_list(linear_feature_columns),
      linear_optimizer = linear_optimizer,
      dnn_feature_columns = ensure_nullable_list(dnn_feature_columns),
      dnn_optimizer = dnn_optimizer,
      dnn_hidden_units = as.integer(dnn_hidden_units),
      dnn_activation_fn = resolve_activation_fn(dnn_activation_fn),
      dnn_dropout = dnn_dropout,
      n_classes = as.integer(n_classes),
      weight_column = weight_column,
      label_vocabulary = label_vocabulary,
      input_layer_partitioner = input_layer_partitioner,
      config = config
    )
  )

  tf_classifier(estimator, "dnn_linear_combined_classifier")
}
