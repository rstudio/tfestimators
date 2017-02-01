#' Construct column placeholders from vectors in an R object
#' @export
tf_auto_inferred_columns <- function(x, columns) {
  layers <- tf$contrib$layers
  lapply(columns, function(column) {
    v <- x[[column]]
    if (is.numeric(v)) {
      layers$real_valued_column(column)
    } else if (is.factor(v)) {
      layers$sparse_column_with_hash_bucket(column)
    } else if (is.character(v)) {
      layers$sparse_column_with_keys(column, keys = levels(v))
    }
  })
}

#' Construct a tflearn Recipe
#'
#' Construct a recipe suitable for use with the higher-level
#' \code{tflearn} modeling routines.
#'
#' @param feature.columns An \R list of tensors, acting as placeholders for
#'   input data.
#' @param input.fn An \R function, returning an \R list binding input data
#'   to the aforementioned feature columns.
#' @param ... Optional named arguments.
#'
#' @family recipes
#' @export
linear_recipe <- function(feature.columns, input.fn, ...) {

  object <- list(
    feature.columns = feature.columns,
    input.fn = input.fn,
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

  feature.columns <- function() {
    tf_auto_inferred_columns(x, features)
  }

  input.fn <- function() {
    feature_columns <- lapply(features, function(feature) {
      tf$constant(x[[feature]])
    })
    names(feature_columns) <- features
    response_column <- tf$constant(x[[response]])
    list(feature_columns, response_column)
  }

  linear_recipe(feature.columns, input.fn)
}


#' @family recipes
#' @export
linear_dnn_combined_recipe <- function(linear.feature.columns, dnn.feature.columns, input.fn, run.config, ...) {
  
  object <- list(
    linear.feature.columns = linear.feature.columns,
    dnn.feature.columns = dnn.feature.columns,
    input.fn = input.fn,
    run.config = run.config,
    ...
  )
  
  class(object) <- "linear_dnn_combined_recipe"
  object
}

# TODO:
#' @title simple_linear_dnn_combined_recipe
#' @name simple_linear_dnn_combined_recipe
#' @export
#' @family recipes
#' @examples
#' # two ways of constructing the same recipe
#' simple_linear_dnn_combined_recipe(mpg ~ cyl, data = mtcars)
#' simple_linear_dnn_combined_recipe(mtcars, response = "mpg", features = c("cyl"))
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
simple_linear_dnn_combined_recipe.default <- function(x, response, linear.features, dnn.features, run.config, ...) {
  
  linear.feature.columns <- function() {
    tf_auto_inferred_columns(x, linear.features)
  }
  
  dnn.feature.columns <- function() {
    tf_auto_inferred_columns(x, dnn.features)
  }
  
  features <- c(linear.features, dnn.features)
  input.fn <- function() {
    feature_columns <- lapply(features, function(feature) {
      tf$constant(x[[feature]])
    })
    names(feature_columns) <- features
    response_column <- tf$constant(x[[response]])
    list(feature_columns, response_column)
  }
  
  run.config <- learn$RunConfig(tf_random_seed=1)
  
  linear_dnn_combined_recipe(linear.feature.columns, dnn.feature.columns, input.fn, run.config)
}
