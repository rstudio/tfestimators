#' Linear Regression
#'
#' Perform linear regression using TensorFlow.
#'
#' @export
#' @family canned estimators
linear_regressor <- function(feature_columns,
                             model_dir = NULL,
                             config = NULL,
                              ...)
{

  # extract feature columns
  feature_columns <- resolve_fn(feature_columns)

  # construct estimator accepting those columns
  lr <- contrib_learn$LinearRegressor(
    feature_columns = feature_columns,
    model_dir = model_dir,
    config = config,
    ...
  )

  tf_model(
    c("linear", "regressor"),
    estimator = lr
  )

}

#' Linear Classification
#'
#' Perform linear classification using TensorFlow.
#' @export
#' @family canned estimators
linear_classifier <- function(feature_columns,
                              n_classes = 2L,
                              model_dir = NULL,
                              config = NULL,
                              ...)
{

  # extract feature columns
  feature_columns <- resolve_fn(feature_columns)

  # construct estimator accepting those columns
  lc <- contrib_learn$LinearClassifier(
    feature_columns = feature_columns,
    n_classes = n_classes,
    model_dir = model_dir,
    config = config,
    ...
  )

  tf_model(
    c("linear", "classifier"),
    estimator = lc
  )
}

