#' Linear regressor model.
#' 
#' Train a linear regression model to predict label value given observation of 
#' feature values.
#' 
#' Input of `fit` and `evaluate` should have following features, otherwise there
#' will be a KeyError:
#' - if isinstance(column, `SparseColumn`): key=column.name, value=a `SparseTensor` 
#' - if isinstance(column, `WeightedSparseColumn`): {key=id column name, value=a `SparseTensor`, 
#'   key=weight column name, value=a `SparseTensor`} 
#' - if isinstance(column, `RealValuedColumn`): key=column.name, value=a `Tensor`
#' 
#' @param feature_columns An iterable containing all the feature columns used by
#'   the model. All items in the set should be instances of classes derived from
#'   `FeatureColumn`.
#' @param model_dir Directory to save model parameters, graph, etc. This can
#'   also be used to load checkpoints from the directory into a estimator to
#'   continue training a previously saved model.
#' @param weight_column_name A string defining feature column name representing
#'   weights. It is used to down weight or boost examples during training. It
#'   will be multiplied by the loss of the example.
#' @param optimizer An instance of `tf.Optimizer` used to train the model. If
#'   `NULL`, will use an Ftrl optimizer.
#' @param gradient_clip_norm A `float` > 0. If provided, gradients are clipped
#'   to their global norm with this clipping ratio. See `tf.clip_by_global_norm`
#'   for more details.
#' @param enable_centered_bias A bool. If TRUE, estimator will learn a centered
#'   bias variable for each class. Rest of the model structure learns the
#'   residual after centered bias.
#' @param label_dimension Number of regression targets per example. This is the
#'   size of the last dimension of the labels and logits `Tensor` objects
#'   (typically, these have shape `[batch_size, label_dimension]`).
#' @param config `RunConfig` object to configure the runtime settings.
#' @param feature_engineering_fn Feature engineering function. Takes features
#'   and labels which are the output of `input_fn` and returns features and
#'   labels which will be fed into the model.
#'   
#' @export
#' @family canned estimators
linear_regressor <- function(
  feature_columns,
  model_dir = NULL,
  weight_column_name = NULL,
  optimizer = NULL,
  gradient_clip_norm = NULL,
  enable_centered_bias = FALSE,
  label_dimension = 1L,
  config = NULL,
  feature_engineering_fn = NULL)
{

  # extract feature columns
  feature_columns <- resolve_feature_columns(feature_columns)

  # construct estimator accepting those columns
  lr <- contrib_learn$LinearRegressor(
    feature_columns = feature_columns,
    model_dir = model_dir,
    weight_column_name = weight_column_name,
    optimizer = optimizer,
    gradient_clip_norm = gradient_clip_norm,
    enable_centered_bias = enable_centered_bias,
    label_dimension = label_dimension,
    config = config,
    feature_engineering_fn = feature_engineering_fn
  )

  tf_model(
    c("linear", "regressor"),
    estimator = lr
  )

}

#' Linear classifier model.
#' 
#' Train a linear model to classify instances into one of multiple possible 
#' classes. When number of possible classes is 2, this is binary classification.
#' 
#' Input of `fit` and `evaluate` should have following features, otherwise there
#' will be a `KeyError`: 
#' * if `weight_column_name` is not `NULL`, a feature with `key=weight_column_name` 
#'   whose value is a `Tensor`.
#' * for each `column` in `feature_columns`: 
#'     - if `column` is a `SparseColumn`, a feature with `key=column.name` whose
#'       `value` is a `SparseTensor`. 
#'     - if `column` is a `WeightedSparseColumn`, two features: the first with 
#'       `key` the id column name, the second with `key` the weight column name. 
#'       Both features' `value` must be a `SparseTensor`. 
#'     - if `column` is a `RealValuedColumn`, a feature with `key=column.name` 
#'       whose `value` is a `Tensor`.
#
#' @param feature_columns An iterable containing all the feature columns used by
#'   the model. All items in the set should be instances of classes derived from
#'   `FeatureColumn`.
#' @param model_dir Directory to save model parameters, graph and etc. This can
#'   also be used to load checkpoints from the directory into a estimator to
#'   continue training a previously saved model.
#' @param n_classes number of label classes. Default is binary classification.
#'   Note that class labels are integers representing the class index (i.e.
#'   values from 0 to n_classes-1). For arbitrary label values (e.g. string
#'   labels), convert to class indices first.
#' @param weight_column_name A string defining feature column name representing
#'   weights. It is used to down weight or boost examples during training. It
#'   will be multiplied by the loss of the example.
#' @param optimizer The optimizer used to train the model. If specified, it
#'   should be either an instance of `tf.Optimizer` or the SDCAOptimizer. If
#'   `NULL`, the Ftrl optimizer will be used.
#' @param gradient_clip_norm A `float` > 0. If provided, gradients are clipped
#'   to their global norm with this clipping ratio. See `tf.clip_by_global_norm`
#'   for more details.
#' @param enable_centered_bias A bool. If TRUE, estimator will learn a centered
#'   bias variable for each class. Rest of the model structure learns the
#'   residual after centered bias.
#' @param config `RunConfig` object to configure the runtime settings.
#' @param feature_engineering_fn Feature engineering function. Takes features
#'   and labels which are the output of `input_fn` and returns features and
#'   labels which will be fed into the model.
#'   
#' @export
#' @family canned estimators
linear_classifier <- function(
  feature_columns,
  model_dir = NULL,
  n_classes = 2L,
  weight_column_name = NULL,
  optimizer = NULL,
  gradient_clip_norm = NULL,
  enable_centered_bias = FALSE,
  config = NULL,
  feature_engineering_fn = NULL)
{

  # extract feature columns
  feature_columns <- resolve_feature_columns(feature_columns)

  # construct estimator accepting those columns
  lc <- contrib_learn$LinearClassifier(
    feature_columns = feature_columns,
    model_dir = model_dir,
    n_classes = as.integer(n_classes),
    weight_column_name = weight_column_name,
    optimizer = optimizer,
    gradient_clip_norm = gradient_clip_norm,
    enable_centered_bias = enable_centered_bias,
    config = config,
    feature_engineering_fn = feature_engineering_fn
  )

  tf_model(
    c("linear", "classifier"),
    estimator = lc
  )
}

