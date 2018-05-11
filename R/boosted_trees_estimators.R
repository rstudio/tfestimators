#' Boosted Trees Estimator
#'
#' Construct a boosted trees estimator.
#'
#' @inheritParams estimators
#'
#' @family canned estimators
#' 
#' @param n_batches_per_layer The number of batches to collect 
#'  statistics per layer.
#' @param n_trees Number trees to be created.
#' @param max_depth Maximum depth of the tree to grow.
#' @param learning_rate Shrinkage parameter to be used when a tree
#'  added to the model.
#' @param l1_regularization Regularization multiplier applied to the
#'  absolute weights of the tree leafs.
#' @param l2_regularization Regularization multiplier applied to the
#'  square weights of the tree leafs.
#' @param tree_complexity Regularization factor to penalize trees
#'  with more leaves.
#' @param min_node_weight Minimum hessian a node must have for a 
#'   split to be considered. The value will be compared with 
#'   sum(leaf_hessian)/(batch_size * n_batches_per_layer).
#'   
#' 
#' @name boosted_trees_estimators
NULL

#' @inheritParams boosted_trees_estimators
#' @name boosted_trees_estimators
#' @export
boosted_trees_regressor <- function(
  feature_columns,
  n_batches_per_layer,
  model_dir = NULL,
  label_dimension = 1L,
  weight_column = NULL,
  n_trees = 100L,
  max_depth = 6L,
  learning_rate = 0.1,
  l1_regularization = 0,
  l2_regularization = 0,
  tree_complexity = 0,
  min_node_weight = 0,
  config = NULL)
{
  if (tensorflow::tf_version() < "1.8.0")
    stop("'boosted_trees_regressor()' requires TensorFlow 1.8+.",
         call. = FALSE)
  
  args <- as.list(environment(), all = TRUE)
  
  estimator <- py_suppress_warnings(
    tf$estimator$BoostedTreesRegressor(
      feature_columns = ensure_nullable_list(feature_columns),
      n_batches_per_layer = ensure_scalar_integer(n_batches_per_layer),
      model_dir = resolve_model_dir(model_dir),
      label_dimension = ensure_scalar_integer(label_dimension),
      weight_column = ensure_scalar_character(weight_column, allow.null = TRUE),
      n_trees = ensure_scalar_integer(n_trees),
      max_depth = ensure_scalar_integer(max_depth),
      learning_rate = ensure_scalar_double(learning_rate),
      l1_regularization = ensure_scalar_double(l1_regularization),
      l2_regularization = ensure_scalar_double(l2_regularization),
      tree_complexity = ensure_scalar_double(tree_complexity),
      min_node_weight = ensure_scalar_double(min_node_weight),
      config = config
    )
  )
  
  new_tf_regressor(estimator, args = args, 
                   subclass = "tf_estimator_regressor_boosted_trees_regressor")
}

#' @inheritParams boosted_trees_estimators
#' @name boosted_trees_estimators
#' @export
boosted_trees_classifier <- function(
  feature_columns,
  n_batches_per_layer,
  model_dir = NULL,
  n_classes = 2L,
  weight_column = NULL,
  label_vocabulary = NULL,
  n_trees = 100L,
  max_depth = 6L,
  learning_rate = 0.1,
  l1_regularization = 0,
  l2_regularization = 0,
  tree_complexity = 0,
  min_node_weight = 0,
  config = NULL)
{
  if (tensorflow::tf_version() < "1.8.0")
    stop("'boosted_trees_classifier()' requires TensorFlow 1.8+.",
         call. = FALSE)
  
  args <- as.list(environment(), all = TRUE)
  
  estimator <- py_suppress_warnings(
    tf$estimator$BoostedTreesClassifier(
      feature_columns = ensure_nullable_list(feature_columns),
      n_batches_per_layer = ensure_scalar_integer(n_batches_per_layer),
      model_dir = resolve_model_dir(model_dir),
      n_classes = ensure_scalar_integer(n_classes),
      weight_column = ensure_scalar_character(weight_column, allow.null = TRUE),
      label_vocabulary = label_vocabulary,
      n_trees = ensure_scalar_integer(n_trees),
      max_depth = ensure_scalar_integer(max_depth),
      learning_rate = ensure_scalar_double(learning_rate),
      l1_regularization = ensure_scalar_double(l1_regularization),
      l2_regularization = ensure_scalar_double(l2_regularization),
      tree_complexity = ensure_scalar_double(tree_complexity),
      min_node_weight = ensure_scalar_double(min_node_weight),
      config = config
    )
  )
  
  new_tf_classifier(estimator, args = args,
                    subclass = "tf_estimator_classifier_boosted_trees_classifier")
}
