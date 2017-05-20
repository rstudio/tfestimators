#' @export
get_input_fn_type <- function(object) {
  is.tf_model(object)
}

#' @export
input_fn <- function(x, ...) {
  UseMethod("input_fn")
}

#' @export
input_fn.default <- function(x, ...) {
  input_fn.data.frame(x, ...)
}

# # TODO: Support unsupervised algorithms
#' @export
input_fn.formula <- function(x, data, ...) {
  parsed <- parse_formula(x)
  input_fn(data, parsed$features, parsed$response, ...)
}

#' Input function constructor for list input
#' 
#' @name input_function_list
NULL

# TODO: Support unsupervised

#' Construct input function from a list object used to feed the estimator.
#' 
#' This is particularly useful when constructing dynamic length of inputs for
#' models like recurrent neural networks.
#' 
#' @param x The list that represents the input source
#' @param features The names of features to be used
#' @param response The response variable name to be used
#' 
#' @examples
#' input_fn1 <- input_fn(
#'    x = list(
#'      feature_names = list(
#'        list(list(1), list(2), list(3)),
#'        list(list(4), list(5), list(6))),
#'      response = list(
#'        list(1, 2, 3), list(4, 5, 6))),
#'    features = c("feature_names"),
#'    response = "response")
#' 
#' @export
#' @family input function constructors
#' @rdname input_function_list
input_fn.list <- function(
  x,
  features,
  response
) {
  validate_input_fn_args(x, features, response)
  function(features_as_named_list = T) {
    if (features_as_named_list) {
      inputs <- tf$constant(
        np$array(
          x$features,
          dtype = np$int64
        )
      )
      labels <- tf$constant(
        np$array(
          x$response
        )
      )
    } else {
      stop("input_fn.list() does not support custom estimator yet")
    }
    list(list(inputs = inputs), labels)
  }
}

#' Input function constructor for rectangular input
#' 
#' @name input_function_rectangular
NULL

#' Construct input function from a data.frame or matrix object used to feed the estimator.
#' 
#' @param x The data.frame or matrix that represents the input source
#' @param features The names of features to be used
#' @param response The response variable name to be used
#' @param batch_size The size of batches
#' @param shuffle Whether to shuffles the queue. Avoid shuffle at prediction time
#' @param num_epochs The number of epochs to iterate over data. If `NULL` will run forever.
#' @param queue_capacity The size of queue to accumulate.
#' @param num_threads The number of threads used for reading and enqueueing. 
#' In order to have predicted and repeatable order of reading and enqueueing,
#' such as in prediction and evaluation mode, `num_threads` should be 1.
#' 
#' @examples
#' features <- c("drat", "cyl")
#' input_fn1 <- input_fn(mtcars, response = "mpg", features = features)
#' 
#' @export
#' @family input function constructors
#' @rdname input_function_rectangular
input_fn.data.frame <-  function(
  x,
  features,
  response = NULL,
  batch_size = 10L,
  shuffle = TRUE,
  num_epochs = 1L,
  queue_capacity = 1000L,
  num_threads = 1L)
{
  validate_input_fn_args(x, features, response)
  num_epochs <- as.integer(num_epochs)
  batch_size <- as.integer(batch_size)
  queue_capacity <- as.integer(queue_capacity)
  num_threads <- as.integer(num_threads)
  # supporting for unsupervised models as well as ingesting data for inference
  input_response <- if(is.null(response)) NULL else as.array(x[,response])
  fn <- function(features_as_named_list) {
    if(features_as_named_list){
      values <- lapply(features, function(feature) {
        as.array(x[, feature])
      })
      names(values) <- features
      fn <- tf$estimator$inputs$numpy_input_fn(
        dict(values),
        input_response,
        batch_size = batch_size,
        shuffle = shuffle,
        num_epochs = num_epochs,
        queue_capacity = queue_capacity,
        num_threads = num_threads)
    } else {
      values <- list(features = data.matrix(x)[,features, drop = FALSE])
      fn <- function(){
        fun <- tf$estimator$inputs$numpy_input_fn(
          values,
          input_response,
          batch_size = batch_size,
          shuffle = shuffle,
          num_epochs = num_epochs,
          queue_capacity = queue_capacity,
          num_threads = num_threads)
        fun <- fun()
        list(
          fun[[1]]$features,
          fun[[2]]
        )
      }
    }
  }
}

#' @export
#' @rdname input_function_rectangular
input_fn.matrix <- function(
  x,
  features,
  response = NULL,
  batch_size = 10L,
  shuffle = TRUE,
  num_epochs = 1L,
  queue_capacity = 1000L,
  num_threads = 1L
) {
  input_fn(as.data.frame(x), features, response, batch_size,
           shuffle, num_epochs, queue_capacity, num_threads)
}

validate_input_fn <- function(input_fn) {
  if (!is.function(input_fn)) stop("input_fn must be a function")
  if (length(formals(input_fn)) != 1) stop("input_fn must contain exactly one argument")
}

#' @export
validate_input_fn_args <- function(x, features, response) {
  ensure_valid_column_names(x, features)
  if (!is.null(response)) {
    ensure_valid_column_names(x, response)
  }
  force(list(x, features, response))
}
