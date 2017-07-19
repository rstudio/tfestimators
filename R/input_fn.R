#' Input function constructor
#' 
#' This function constructs input function from various types of input used to feed the
#' estimator.
#' 
#' For list objects, this method is particularly useful when constructing
#' dynamic length of inputs for models like recurrent neural networks. Note that
#' some arguments are not available yet for input_fn applied to list objects. 
#' See S3 method signatures below for more details.
#' 
#' @param object The object that represents the input source
#' @param features The names of features to be used
#' @param response The response variable name to be used
#' @param batch_size The size of batches
#' @param shuffle Whether to shuffles the queue. Avoid shuffle at prediction 
#'   time
#' @param num_epochs The number of epochs to iterate over data. If `NULL` will 
#'   run forever.
#' @param queue_capacity The size of queue to accumulate.
#' @param num_threads The number of threads used for reading and enqueueing. In 
#'   order to have predicted and repeatable order of reading and enqueueing, 
#'   such as in prediction and evaluation mode, `num_threads` should be 1.
#'
#' @name input_fn
#' @family input functions
NULL


#' @export
input_fn <- function(object, ...) {
  UseMethod("input_fn")
}

#' @export
#' @rdname input_fn
input_fn.default <- function(object, ...) {
  input_fn.data.frame(as.data.frame(object), ...)
}

#' @export
#' @rdname input_fn
#' @examples 
#' # Construct the input function through formula interface
#' input_fn1 <- input_fn(mpg ~ drat + cyl, mtcars)
#' 
input_fn.formula <- function(object, data, ...) {
  parsed <- parse_formula(object)
  input_fn(data, parsed$features, parsed$response, ...)
}

#' 
#' @examples
#' # Construct the input function from a list object
#' input_fn1 <- input_fn(
#'    object = list(
#'      feature1 = list(
#'        list(list(1), list(2), list(3)),
#'        list(list(4), list(5), list(6))),
#'      feature2 = list(
#'        list(list(7), list(8), list(9)),
#'        list(list(10), list(11), list(12))),
#'      response = list(
#'        list(1, 2, 3), list(4, 5, 6))),
#'    features = c("feature1", "feature2"),
#'    response = "response",
#'    batch_size = 10L)
#' 
#' @export
#' @family input function constructors
#' @rdname input_fn
input_fn.list <- function(
  object,
  features,
  response = NULL,
  batch_size = 10L,
  shuffle = TRUE,
  num_epochs = 1L,
  queue_capacity = 1000L,
  num_threads = 1L
) {
  all_names <- object_names(object)
  features <- vars_select(all_names, !! enquo(features))
  if (!missing(response))
    response <- vars_select(all_names, !! enquo(response))

  num_epochs <- as.integer(num_epochs)
  batch_size <- as.integer(batch_size)
  queue_capacity <- as.integer(queue_capacity)
  num_threads <- as.integer(num_threads)
  
  # Support for unsupervised models as well as ingesting data for inference
  if (is.null(response)) {
    input_response <- NULL
  } else {
    input_response <- object$response
    names(input_response) <- NULL
    input_response <- np$array(input_response)
  }

  function(features_as_named_list) {
    if (features_as_named_list) {
      features_dict <- dict()
      lapply(features, function(feature){
        features_dict[[feature]] <- np$array(
          object[[feature]],
          dtype = np$int64
        )
      })
      fn <- tf$estimator$inputs$numpy_input_fn(
        features_dict,
        input_response,
        batch_size = batch_size,
        shuffle = shuffle,
        num_epochs = num_epochs,
        queue_capacity = queue_capacity,
        num_threads = num_threads)
      fn
    } else {
      features_list <- lapply(features, function(feature) object[[feature]])
      names(features_list) <- NULL
      features_dict <- dict()
      features_dict$features <- np$array(
        features_list,
        dtype = np$int64
      )
      fn <- function(){
        fun <- tf$estimator$inputs$numpy_input_fn(
          features_dict,
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


#'   
#' @examples
#' # Construct the input function from a data.frame object
#' input_fn1 <- input_fn(mtcars, response = mpg, features = c(drat, cyl))
#' 
#' @export
#' @family input function constructors
#' @rdname input_fn
input_fn.data.frame <-  function(
  object,
  features,
  response = NULL,
  batch_size = 10L,
  shuffle = TRUE,
  num_epochs = 1L,
  queue_capacity = 1000L,
  num_threads = 1L)
{
  all_names <- object_names(object)
  features <- vars_select(all_names, !! enquo(features))
  if (!missing(response))
    response <- vars_select(all_names, !! enquo(response))
  
  num_epochs <- as.integer(num_epochs)
  batch_size <- as.integer(batch_size)
  queue_capacity <- as.integer(queue_capacity)
  num_threads <- as.integer(num_threads)
  # Support for unsupervised models as well as ingesting data for inference
  input_response <- if (is.null(response)) NULL else as.array(object[, response])
  fn <- function(features_as_named_list) {
    if (features_as_named_list) {
      values <- lapply(features, function(feature) {
        as.array(object[, feature])
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
      fn
    } else {
      values <- list(features = data.matrix(object)[,features, drop = FALSE])
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
#' @rdname input_fn
input_fn.matrix <- function(
  object,
  features,
  response = NULL,
  batch_size = 10L,
  shuffle = TRUE,
  num_epochs = 1L,
  queue_capacity = 1000L,
  num_threads = 1L
) {
  if (is.null(colnames(object)))
    stop("You must provide colnames in order to create an input_fn from a matrix")
  
  all_names <- object_names(object)
  features <- vars_select(all_names, !! enquo(features))
  if (!missing(response))
    response <- vars_select(all_names, !! enquo(response))
  
  input_fn.data.frame(object, features, response, batch_size,
           shuffle, num_epochs, queue_capacity, num_threads)
}

#' Construct input function that would feed dict of numpy arrays into the model.
#' 
#' This returns a function outputting `features` and `target` based on the dict 
#' of numpy arrays. The dict `features` has the same keys as the `x`.
#' 
#' @param x dict of numpy array object.
#' @param y numpy array object. `NULL` if absent.
#' @param batch_size Integer, size of batches to return.
#' @param num_epochs Integer, number of epochs to iterate over data. If `NULL`
#'   will run forever.
#' @param shuffle Boolean, if TRUE shuffles the queue. Avoid shuffle at
#'   prediction time.
#' @param queue_capacity Integer, size of queue to accumulate.
#' @param num_threads Integer, number of threads used for reading and
#'   enqueueing. In order to have predicted and repeatable order of reading and
#'   enqueueing, such as in prediction and evaluation mode, `num_threads` should
#'   be 1. #'
#' @section Raises: ValueError: if the shape of `y` mismatches the shape of
#'   values in `x` (i.e., values in `x` have same shape). TypeError: `x` is not
#'   a dict or `shuffle` is not bool.
#'   
#' @export
#' @family input functions
numpy_input_fn <- function(x, y = NULL, batch_size = 128L, num_epochs = 1L, shuffle = NULL, queue_capacity = 1000L, num_threads = 1L) {
  function(features_as_named_list) {
    tf$estimator$inputs$numpy_input_fn(
      x = x,
      y = y,
      batch_size = as.integer(batch_size),
      num_epochs = as.integer(num_epochs),
      shuffle = shuffle,
      queue_capacity = as.integer(queue_capacity),
      num_threads = as.integer(num_threads)
    )
  }
}

# input functions take zero arguments however on the R side we allow input functions
# with a single boolean argument that determines whether features should be provided
# as a named list. this function validates the input_fn and normalizes it into a 
# no-argument function if necessary.
#
# this functionality is provided as the expected input type differs based on
# whether a TensorFlow 'canned estimator' is used, or a user-defined 'custom estimator'
# is used.
#
# note that the 'input_fn' accepted as input may either be itself an input function,
# or a function that returns an 'input_fn', that function accepting a single argument
# to determine whether it should return data as a dictionary or a plain tensor.
normalize_input_fn <- function(object, input_fn) {
  
  if (!is.function(input_fn)) 
    stop("input_fn must be a function", call. = FALSE)
  
  nargs <- length(formals(input_fn))
  
  # if the input function doesn't accept any arguments, assume
  # that it's already an 'input_fn' as expected by TensorFlow
  if (nargs == 0)
    return(input_fn)
  
  # if the input function accepts a single argument, assume that
  # it should be used to generate and provide an input function
  if (nargs == 1) {
    custom <- inherits(object, "tf_custom_estimator")
    return(input_fn(!custom))
  }
  
  # other function signatures are errors
  stop("'input_fn' should accept 0 or 1 arguments")
}

