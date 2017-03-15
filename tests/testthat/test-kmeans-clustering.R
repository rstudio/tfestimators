context("Testing kmeans clustering")

# WIP - need to revisit the issue in KMeansClustering
test_that("kmeans_clustering() runs successfully", {

  construct_new_input_fn <-  function(
    x,
    features,
    response = NULL,
    feature_as_named_list = TRUE,
    id_column = NULL)
  {
    force(list(x, response, features))
    function() {
      if (feature_as_named_list) {
        # For linear and dnn we have to do this due to nature of feature columns
        feature_columns <- lapply(features, function(feature) {
          tf$constant(x[[feature]])
        })
        names(feature_columns) <- features
      } else {
        # This works for custom model
        # TODO: Consider a separate spec constructor
        feature_columns <- tf$constant(as.matrix(x[, features]))
      }
      if (!is.null(response)) {
        response_column <- tf$constant(x[[response]])
        return(list(feature_columns, response_column))
      } else {
        return(list(feature_columns, NULL))
      }
    }
  }

  constructed_input_fn <- construct_new_input_fn(
    mtcars,
    features = c("drat", "cyl", "qsec"),
    feature_as_named_list = F)
  run_options <- run_options()
  kmeans <- learn$KMeansClustering(
    num_clusters = 3,
    model_dir = run_options$model_dir)
  # kmeans$fit(input_fn = constructed_input_fn, steps = 1L)
  # kmeans$score(input_fn = constructed_input_fn)
})
