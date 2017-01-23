#' TensorFlow -- Linear Regression
#'
#' Perform linear regression using TensorFlow.
#'
#' @template roxlate-tf-x
#' @template roxlate-tf-response
#' @template roxlate-tf-features
#' @template roxlate-tf-intercept
#' @template roxlate-input-fn
#' @template roxlate-tf-dots
#' @template roxlate-tf-options
#'
#' @export
tf_linear_regression <- function(x,
                                 response,
                                 features,
                                 intercept = TRUE,
                                 input.fn = NULL,
                                 tf.options = tf_options(),
                                 ...)
{
  tf_backwards_compatibility_api()
  tf_prepare_response_features_intercept(x, response, features, intercept)

  # Construct TF.Learn columns
  columns <- tf_columns(x, features)
  
  # Construct regressor accepting those columns
  lr <- learn$LinearRegressor(
    feature_columns = columns,
    optimizer = tf.options$optimizer
  )
  
  
  # Define input function (supplying data to aforementioned
  # feature placeholders, as well as response)
  input_fn <- input.fn %||% function(dataset) {
    
    # Define feature columns
    feature_columns <- lapply(features, function(feature) {
      tf$constant(dataset[[feature]])
    })
    names(feature_columns) <- features
    
    # Define response column
    response_column <- tf$constant(dataset[[response]])
    
    # Return as two-element list
    list(feature_columns, response_column)
    
  }
  
  # Run the model
  lr$fit(
    input_fn = function() { input_fn(x) },
    steps = tf.options$steps
  )
  
  # TODO: extract some information relevant to the R user
  # (coefficients?)
  
  tf_model(
    "linear_regression",
    estimator = lr,
    input_fn = input_fn
  )

}
