#' Support Vector Machine (SVM) model for binary classification.
#' 
#' Currently, only linear SVMs are supported. For the underlying optimization
#' problem, the `SDCAOptimizer` is used. For performance and convergence tuning,
#' the num_loss_partitions parameter passed to `SDCAOptimizer` (see `__init__()`
#' method), should be set to (#concurrent train ops per worker) x (#workers). If
#' num_loss_partitions is larger or equal to this value, convergence is
#' guaranteed but becomes slower as num_loss_partitions increases. If it is set
#' to a smaller value, the optimizer is more aggressive in reducing the global
#' loss but convergence is not guaranteed. The recommended value in tf.learn
#' (where there is one process per worker) is the number of workers running the
#' train steps. It defaults to 1 (single machine).
#' 
#' @param example_id_column A string defining the feature column name
#'   representing example ids. Used to initialize the underlying optimizer.
#' @param feature_columns An iterable containing all the feature columns used by
#'   the model. All items in the set should be instances of classes derived from
#'   `FeatureColumn`.
#' @param weight_column_name A string defining feature column name representing
#'   weights. It is used to down weight or boost examples during training. It
#'   will be multiplied by the loss of the example.
#' @param model_dir Directory to save model parameters, graph and etc. This can
#'   also be used to load checkpoints from the directory into a estimator to
#'   continue training a previously saved model.
#' @param l1_regularization L1-regularization parameter. Refers to global L1
#'   regularization (across all examples).
#' @param l2_regularization L2-regularization parameter. Refers to global L2
#'   regularization (across all examples).
#' @param num_loss_partitions number of partitions of the (global) loss function
#'   optimized by the underlying optimizer (SDCAOptimizer).
#' @param kernels A list of kernels for the SVM. Currently, no kernels are
#'   supported. Reserved for future use for non-linear SVMs.
#' @param config RunConfig object to configure the runtime settings.
#' @param feature_engineering_fn Feature engineering function. Takes features
#'   and labels which are the output of `input_fn` and returns features and
#'   labels which will be fed into the model.
#' 
svm_classifier <- function(example_id_column,
                           feature_columns,
                           weight_column_name = NULL,
                           model_dir = NULL,
                           l1_regularization = 0.0,
                           l2_regularization = 0.0,
                           num_loss_partitions = 1L,
                           kernels = NULL,
                           config = NULL,
                           feature_engineering_fn = NULL)
{
  estimator <- tf$contrib$learn$SVM(
    example_id_column = example_id_column,
    feature_columns = feature_columns,
    weight_column_name = weight_column_name,
    model_dir = model_dir,
    l1_regularization = l1_regularization,
    l2_regularization = l2_regularization,
    num_loss_partitions = num_loss_partitions,
    kernels = kernels,
    config = config,
    feature_engineering_fn = feature_engineering_fn
  )
  
  tf_classifier(estimator, "svm")
}
