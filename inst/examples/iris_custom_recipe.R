library(tflearn)

setup_shortcuts()

temp_model_dir <- tempfile()
dir.create(temp_model_dir)

iris_data <- learn_datasets_lib$load_dataset("iris")

feature_names <- c("V1", "V2", "V3", "V4")
iris_features <- as.data.frame(iris_data$data)
colnames(iris_features) <- feature_names
iris_labels <- iris_data$target

feature_columns <- lapply(feature_names, function(colname) {
  contrib_layers_lib$real_valued_column(colname)
})

iris_input_fn <- function() {
  features <- lapply(feature_names, function(feature_name) {
    tf$constant(iris_features[[feature_name]])
  })
  names(features) <- feature_names
  labels <- tf$constant(iris_labels)
  return(list(features, labels))
}

config <- run_config_lib$RunConfig(tf_random_seed=1)

classifier <- tf$contrib$learn$DNNClassifier(
  feature_columns = feature_columns,
  hidden_units = c(10L, 15L, 10L),
  n_classes = 3L,
  model_dir = temp_model_dir,
  config = config)

classifier$fit(input_fn = iris_input_fn, steps = 2)

predictions <- classifier$predict(input_fn = iris_input_fn)
predictions <- iterate(predictions)


