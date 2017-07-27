#' Construct an Input Function
#' 
#' This function constructs input function from various types of input used to
#' feed different TensorFlow estimators.
#' 
#' For list objects, this method is particularly useful when constructing
#' dynamic length of inputs for models like recurrent neural networks. Note that
#' some arguments are not available yet for input_fn applied to list objects. 
#' See S3 method signatures below for more details.
#' 
#' @param object,data An 'input source' -- either a data set (e.g. an \R `data.frame`),
#'   or another kind of object that can provide the data required for training.
#' @param features The names of feature variables to be used.
#' @param response The name of the response variable.
#' @param batch_size The batch size.
#' @param shuffle Whether to shuffle the queue. When \code{"auto"} (the default),
#'   shuffling will be performed except when this input function is called by
#'   a \code{predict()} method.
#' @param num_epochs The number of epochs to iterate over data.
#' @param queue_capacity The size of queue to accumulate.
#' @param num_threads The number of threads used for reading and enqueueing. In 
#'   order to have predictable and repeatable order of reading and enqueueing, 
#'   such as in prediction and evaluation mode, `num_threads` should be 1.
#' @param ... Optional arguments passed on to implementing submethods.
#'
#' @family input functions
#' @export
input_fn <- function(object, ...) {
  UseMethod("input_fn")
}

#' @export
#' @rdname input_fn
input_fn.default <- function(object, ...) {
  input_fn(as.data.frame(object, stringsAsFactors = FALSE), ...)
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

#' @examples
#' # Construct the input function from a data.frame object
#' input_fn1 <- input_fn(mtcars, response = mpg, features = c(drat, cyl))
#' 
#' @export
#' @family input function constructors
#' @rdname input_fn
input_fn.data.frame <- function(object,
                                features,
                                response = NULL,
                                batch_size = 128,
                                shuffle = "auto",
                                num_epochs = 1,
                                queue_capacity = 1000,
                                num_threads = 1,
                                ...)
{
  all_names <- object_names(object)
  features <- vars_select(all_names, !! enquo(features))
  if (!missing(response))
    response <- vars_select(all_names, !! enquo(response))
  
  num_epochs <- as_nullable_integer(num_epochs)
  batch_size <- as.integer(batch_size)
  queue_capacity <- as.integer(queue_capacity)
  num_threads <- as.integer(num_threads)
  
  # convert 'shuffle' at runtime based on call context
  resolve_shuffle <- function(shuffle) {
    
    if (!identical(shuffle, "auto"))
      return(shuffle)
    
    # check to see if we're being called by a predict method
    calls <- sys.calls()
    match <- Find(function(call) {
      identical(call[[1]], quote(predict.tf_estimator)) ||
      identical(call[[1]], quote(object$estimator$predict))
    }, calls)
    
    # prefer shuffling if we're not within predict
    is.null(match)
  }
  
  # coerce vectors to a TensorFlow-friendly format when appropriate
  coerce <- function(variable) {
    
    # convert lists to numpy arrays
    if (is.list(variable))
      return(np$array(unname(variable), dtype = np$int64))
    
    # convert factors to [0, n] range
    if (is.factor(variable))
      variable <- as.integer(variable) - 1L
    
    as.array(variable)
  }
  
  # determine response variable
  input_response <- if (!is.null(response))
    coerce(object[[response]])
  
  # input function to be used with canned estimators
  canned_input_fn_generator <- function() {
    
    # convert to named R list
    values <- (function() {
      
      result <- lapply(features, function(feature) {
        coerce(object[[feature]])
      })
      names(result) <- features
      return(dict(result))
      
    })()
    
    
    # generate numpy-style input function
    tf$estimator$inputs$numpy_input_fn(
      values,
      input_response,
      batch_size = batch_size,
      shuffle = shuffle,
      num_epochs = num_epochs,
      queue_capacity = queue_capacity,
      num_threads = num_threads
    )
  }
  
  # input function to be used with custom estimators
  custom_input_fn_generator <- function() {
    
    values <- (function() {
      
      # TODO: since we're creating an R matrix, this implies that
      # all features must have the same data type?
      return(list(features = data.matrix(object[features])))
      
    })()
    
    # return R function that provides list of features + input function
    function() {
      
      input <- tf$estimator$inputs$numpy_input_fn(
        values,
        input_response,
        batch_size = batch_size,
        shuffle = shuffle,
        num_epochs = num_epochs,
        queue_capacity = queue_capacity,
        num_threads = num_threads
      )()
      
      list(
        input[[1]]$features,
        input[[2]]
      )
    }
  }
  
  # return function which provides canned vs custom input function
  # as requested
  function(estimator) {
    shuffle <<- resolve_shuffle(shuffle)
    if (inherits(estimator, "tf_custom_estimator"))
      return(custom_input_fn_generator())
    else
      return(canned_input_fn_generator())
  }
}

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
input_fn.list <- input_fn.data.frame

#' @export
#' @rdname input_fn
input_fn.matrix <- function(object, ...)
{
  if (is.null(colnames(object)))
    stop("cannot create input function from matrix without column names")
  
  input_fn(
    as.data.frame(object, stringsAsFactors = FALSE),
    ...
  )
}

#' Construct input function that would feed dict of numpy arrays into the model.
#' 
#' This returns a function outputting `features` and `target` based on the dict 
#' of numpy arrays. The dict `features` has the same keys as the `x`.
#'
#' Note that this function is still experimental and should only be used if
#' necessary, e.g. feed in data that's dictionary of numpy arrays.
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
numpy_input_fn <- function(x,
                           y = NULL,
                           batch_size = 128,
                           num_epochs = 1,
                           shuffle = NULL,
                           queue_capacity = 1000,
                           num_threads = 1)
{
  function(estimator) {
    tf$estimator$inputs$numpy_input_fn(
      x = x,
      y = y,
      batch_size = as.integer(batch_size),
      num_epochs = as_nullable_integer(num_epochs),
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
  if (nargs == 1)
    return(input_fn(object))
  
  # other function signatures are errors
  stop("'input_fn' should accept 0 or 1 arguments")
}
