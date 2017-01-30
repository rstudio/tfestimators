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

tf_setting <- function(name, default) {

  # Check for environment variable with associated name
  env <- toupper(gsub(".", "_", name, fixed = TRUE))
  val <- Sys.getenv(env, unset = NA)
  if (!is.na(val))
    return(val)

  # Check for R option with associated name
  val <- getOption(name)
  if (!is.null(val))
    return(val)

  # Use default value
  default
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
tf_recipe <- function(feature.columns, input.fn, ...) {

  object <- list(
    feature.columns = feature.columns,
    input.fn = input.fn,
    ...
  )

  class(object) <- "tf_recipe"
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
#' tf_simple_recipe(mpg ~ cyl, data = mtcars)
#' tf_simple_recipe(mtcars, response = "mpg", features = c("cyl"))
tf_simple_recipe <- function(x, ...) {
  UseMethod("tf_simple_recipe")
}

#' @export
tf_simple_recipe.formula <- function(x, data, ...) {
  parsed <- parse_formula(x)
  tf_simple_recipe(data, parsed$response, parsed$features)
}

#' @export
tf_simple_recipe.default <- function(x, response, features, ...) {

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

  tf_recipe(feature.columns, input.fn)
}
