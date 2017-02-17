#' Construct column placeholders from vectors in an R object
#' @export
tf_auto_inferred_columns <- function(x, columns) {
  layers <- tf$contrib$layers
  lapply(columns, function(column) {
    v <- x[[column]]
    if (is.numeric(v)) {
      layers$real_valued_column(column)
    } else if (is.factor(v)) {
      layers$sparse_column_with_keys(column, keys = levels(v))
    } else if (is.character(v)) {
      layers$sparse_column_with_hash_bucket(column)
    }
  })
}

#' @export
tf_simple_input_fn <-  function(x, response, features, feature_as_named_list = TRUE, id_column = NULL) {
  if (!is.null(id_column)) {
    x[id_column] <- 1:nrow(x)
    features <- c(features, id_column)
    # TODO: Support custom id_column function
  }
  force(list(x, response, features))
  function(newdata = NULL) {
    if (!is.null(newdata))
      x <<- newdata
    if (feature_as_named_list) {
      # For linear and dnn we have to do this due to nature of feature columns
      feature_columns <- lapply(features, function(feature) {
        tf$constant(x[[feature]])
      })
      names(feature_columns) <- features
    } else {
      # This works for custom model
      feature_columns <- tf$constant(as.matrix(x[, features]))
    }
    response_column <- tf$constant(x[[response]])
    list(feature_columns, response_column)
  }
}

#' @family recipes
#' @export
custom_model_recipe <- function(model_fn,
                                input_fn,
                                ...)
{
  object <- list(
    model_fn = model_fn,
    input_fn = input_fn,
    ...
  )

  class(object) <- "custom_model_recipe"
  object
}

#' @family recipes
#' @export
simple_custom_model_recipe <- function(x, ...) {
  UseMethod("simple_custom_model_recipe")
}

#' @family recipes
#' @export
simple_custom_model_recipe.default <- function(x,
                                               response,
                                               features,
                                               model_fn,
                                               ...)
{
  input_fn <- tf_simple_input_fn(x, response, features,
                                 feature_as_named_list = FALSE)
  custom_model_recipe(model_fn, input_fn)
}

#' @export
simple_custom_model_recipe.formula <- function(x, data, model_fn, ...) {
  parsed <- parse_formula(x)
  simple_custom_model_recipe(data, parsed$response, parsed$features, model_fn)
}

#' @family recipes
#' @export
svm_recipe <- function(feature_columns,
                       input_fn,
                       example_id_column,
                       weight_column_name = NULL,
                       ...)
{
  object <- list(
    feature_columns = feature_columns,
    input_fn = input_fn,
    example_id_column = example_id_column,
    weight_column_name = weight_column_name,
    ...
  )

  class(object) <- "svm_recipe"
  object
}

#' @family recipes
#' @export
simple_svm_recipe <- function(x, ...) {
  UseMethod("simple_svm_recipe")
}

#' @family recipes
#' @export
simple_svm_recipe.default <- function(x,
                                      response,
                                      features,
                                      ...)
{
  feature_columns <- function() {
    tf_auto_inferred_columns(x, features)
  }

  input_fn <- tf_simple_input_fn(x, response, features, id_column = "id_column")
  svm_recipe(feature_columns, input_fn, example_id_column = "id_column", weight_column_name = NULL)
}

#' @export
simple_svm_recipe.formula <- function(x, data, ...) {
  parsed <- parse_formula(x)
  simple_svm_recipe(data, parsed$response, parsed$features)
}


#' Construct a tflearn Recipe
#'
#' Construct a recipe suitable for use with the higher-level
#' \code{tflearn} modeling routines.
#'
#' @param feature_columns An \R list of tensors, acting as placeholders for
#'   input data.
#' @param input_fn An \R function, returning an \R list binding input data
#'   to the aforementioned feature columns.
#' @param ... Optional named arguments.
#'
#' @family recipes
#' @export
linear_recipe <- function(feature_columns, input_fn, ...) {

  object <- list(
    feature_columns = feature_columns,
    input_fn = input_fn,
    ...
  )

  class(object) <- "linear_recipe"
  object
}

#' Construct a Simple tflearn Recipe
#'
#' Construct a simple recipe suitable for use with the higher-level
#' \code{tflearn} modeling routines. This can be used to directly model a
#' response variable as a function of a set of untransformed features in a
#' dataset.
#'
#' @param x An \R object; typically a \code{data.frame} or a \code{formula}. See
#'   examples for usage.
#' @param ... Optional arguments passed to implementing methods.
#'
#' @family recipes
#' @export
#'
#' @examples
#' # two ways of constructing the same recipe
#' simple_linear_recipe(mpg ~ cyl, data = mtcars)
#' simple_linear_recipe(mtcars, response = "mpg", features = c("cyl"))
simple_linear_recipe <- function(x, ...) {
  UseMethod("simple_linear_recipe")
}

#' @export
simple_linear_recipe.formula <- function(x, data, ...) {
  parsed <- parse_formula(x)
  simple_linear_recipe(data, parsed$response, parsed$features)
}

#' @export
simple_linear_recipe.default <- function(x, response, features, ...) {

  feature_columns <- function() {
    tf_auto_inferred_columns(x, features)
  }

  input_fn <- tf_simple_input_fn(x, response, features)

  linear_recipe(feature_columns, input_fn)
}


#' @family recipes
#' @export
linear_dnn_combined_recipe <- function(linear_feature_columns,
                                       dnn_feature_columns,
                                       input_fn,
                                       ...)
{
  object <- list(
    linear_feature_columns = linear_feature_columns,
    dnn_feature_columns = dnn_feature_columns,
    input_fn = input_fn,
    ...
  )
  
  class(object) <- "linear_dnn_combined_recipe"
  object
}

#' Simple Linear DNN Combined Recipe
#' 
#' TODO: Description
#' 
#' @title simple_linear_dnn_combined_recipe
#' @name simple_linear_dnn_combined_recipe
#' @export
#' @family recipes
#' @examples
#' ## # two ways of constructing the same recipe
#' ## simple_linear_dnn_combined_recipe(mpg ~ cyl, data = mtcars)
#' ## simple_linear_dnn_combined_recipe(mtcars, response = "mpg", features = c("cyl"))
simple_linear_dnn_combined_recipe <- function(x, ...) {
  UseMethod("simple_linear_dnn_combined_recipe")
}

# TODO: Interface like simple_linear_dnn_combined_recipe(mpg ~ linear(cyl) + dnn(drat) )
#' #' @export
#' simple_linear_dnn_combined_recipe.formula <- function(x, data, ...) {
#'   parsed <- parse_formula(x)
#'   simple_linear_dnn_combined_recipe(data, parsed$response, parsed$features)
#' }

#' @export
simple_linear_dnn_combined_recipe.default <- function(x,
                                                      response,
                                                      linear_features,
                                                      dnn_features,
                                                      ...) 
{
  linear_feature_columns <- function() {
    tf_auto_inferred_columns(x, linear_features)
  }
  
  dnn_feature_columns <- function() {
    tf_auto_inferred_columns(x, dnn_features)
  }
  
  features <- c(linear_features, dnn_features)
  input_fn <- tf_simple_input_fn(x, response, features)
  
  linear_dnn_combined_recipe(linear_feature_columns, dnn_feature_columns, input_fn)
}
