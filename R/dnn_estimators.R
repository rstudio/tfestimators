#' Deep Neural Networks
#'
#' Create a deep neural network (DNN) estimator.
#'
#' @inheritParams estimators
#' 
#' @template roxlate-activation-fn
#' @templateVar name activation_fn
#' @templateVar default relu
#' 
#' @param hidden_units An integer vector, indicating the number of hidden
#'   units in each layer. All layers are fully connected. For example,
#'   `c(64, 32)` means the first layer has 64 nodes, and the second layer
#'   has 32 nodes.
#' @param optimizer Either the name of the optimizer to be used when training
#'   the model, or a TensorFlow optimizer instance. Defaults to the Adagrad
#'   optimizer.
#' @param dropout When not `NULL`, the probability we will drop out a given
#'   coordinate.
#'
#' @family canned estimators
#' @name dnn_estimators
NULL

#' @inheritParams dnn_estimators
#' @name dnn_estimators
#' @export
dnn_regressor <- function(hidden_units,
                          feature_columns,
                          model_dir = NULL,
                          label_dimension = 1L,
                          weight_column = NULL,
                          optimizer = "Adagrad",
                          activation_fn = "relu",
                          dropout = NULL,
                          input_layer_partitioner = NULL,
                          config = NULL)
{
  args <- as.list(environment(), all = TRUE)
  
  estimator <- py_suppress_warnings(
    tf$estimator$DNNRegressor(
      hidden_units = lapply(hidden_units, ensure_scalar_integer),
      feature_columns = ensure_nullable_list(feature_columns),
      model_dir = resolve_model_dir(model_dir),
      label_dimension = ensure_scalar_integer(label_dimension),
      weight_column = ensure_scalar_character(weight_column, allow.null = TRUE),
      optimizer = optimizer,
      activation_fn = resolve_activation_fn(activation_fn),
      dropout = ensure_scalar_double(dropout, allow.null = TRUE),
      input_layer_partitioner = input_layer_partitioner,
      config = config
    )
  )

  new_tf_regressor(estimator, args = args,
                   subclass = "tf_estimator_regressor_dnn_regressor")
}

#' @inheritParams dnn_estimators
#' @name dnn_estimators
#' @export
dnn_classifier <- function(hidden_units,
                           feature_columns,
                           model_dir = NULL,
                           n_classes = 2L,
                           weight_column = NULL,
                           label_vocabulary = NULL,
                           optimizer = "Adagrad",
                           activation_fn = "relu",
                           dropout = NULL,
                           input_layer_partitioner = NULL,
                           config = NULL)
{
  args <- as.list(environment(), all = TRUE)
  
  estimator <- py_suppress_warnings(
    tf$estimator$DNNClassifier(
      hidden_units = lapply(hidden_units, ensure_scalar_integer),
      feature_columns = ensure_nullable_list(feature_columns),
      model_dir = resolve_model_dir(model_dir),
      n_classes = ensure_scalar_integer(n_classes),
      weight_column = ensure_scalar_character(weight_column, allow.null = TRUE),
      label_vocabulary = label_vocabulary,
      optimizer = optimizer,
      activation_fn = resolve_activation_fn(activation_fn),
      dropout = ensure_scalar_double(dropout, allow.null = TRUE),
      input_layer_partitioner = input_layer_partitioner,
      config = config
    )
  )

  new_tf_classifier(estimator, args = args, 
                    subclass = "tf_estimator_classifier_dnn_classifier")
}

