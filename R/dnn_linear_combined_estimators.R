#' Linear Combined Deep Neural Networks
#' 
#' Also known as \code{wide-n-deep} estimators, these are estimators for
#' TensorFlow Linear and DNN joined models for regression.
#' 
#' @inheritParams estimators
#' 
#' @param linear_feature_columns An iterable containing all the feature columns 
#'   used by linear part of the model. All items in the set must be instances of
#'   classes derived from `FeatureColumn`.
#' @param linear_optimizer An instance of `tf.Optimizer` used to apply gradients
#'   to the linear part of the model. Defaults to FTRL optimizer.
#' @param dnn_feature_columns An iterable containing all the feature columns 
#'   used by deep part of the model. All items in the set must be instances of 
#'   classes derived from `FeatureColumn`.
#' @param dnn_optimizer An instance of `tf.Optimizer` used to apply gradients to
#'   the deep part of the model. Defaults to Adagrad optimizer.
#' @param dnn_hidden_units List of hidden units per layer. All layers are fully 
#'   connected.
#' @param dnn_activation_fn Activation function applied to each layer. If NULL, 
#'   will use `tf$nn$relu`.
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
                                          dnn_activation_fn = tf$nn$relu,
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
      dnn_activation_fn = dnn_activation_fn,
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
                                           dnn_activation_fn = tf$nn$relu,
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
      dnn_activation_fn = dnn_activation_fn,
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
