# Architecture and Main Components of This Package

## Spec Constructors

Spec constructors are for constructing the input and features for a particular estimator. All estimators require input_fn. Canned estimators, in addition, requires specification for feature columns. 

The first spec constructor is feature columns required for canned estimators such as `DNNEstimator`. This specifies the feature transformations and combinations for a model, e.g. `column_embedding` that converts a categorical variable into embedding and `column_crossed` that combines two variables in a specified way. 

Users can use the default `feature_columns` function to convert columns in an automatic fashion without any fancy feature engineering, e.g. numeric variables are converted using `column_real_valued()`, factor variables are converted using `column_with_keys()`, and character variables are converted using `column_with_hash_bucket()`. 

``` r
fcs <- feature_columns(mtcars, c("drat", "cyl"))
```

Users can also write their own custom feature columns transformation function like the following that transforms different columns in a `lapply` loop. 

``` r
custom_feature_columns <- function(x, columns) {
  ensure_valid_column_names(x, columns)
  function() {
    lapply(columns, function(column_name) {
      column_values <- x[[column_name]]
      if (column_name == "profit") {
        column_real_valued(column_name)
      } else if (column_name == "profession") {
        column_with_keys(column_name)
      } else {
      	...
      }
    })
  }
}
```

Another spec constructor is the input_fn required for the estimators. This is where users provide an input source, e.g. in-memory dataframe or matrix, streaming data, serialized data formats, etc. 

Users have two ways to specify in-memory data set - using formula interface or passing `features` and `response` arguments. Note that there's an argument named `features_as_named_list` that should be `TRUE` if this is used by a canned estimator and should be `FALSE` if this is used by a custom estimator. 

``` r
input_fn.default(mtcars, response = "mpg", features = c("drat", "cyl"))
input_fn.formula(mpg ~ drat + cyl, data = mtcars)

```

Users can also write custom function to convert each feature into a `Tensor` or `SparseTensor` according to the needs, e.g. a function called `custom_function`. This function should return a list that consists of `input_fn` and `features_as_named_list` so the custom or canned estimator can recognize them. The following code has a few places commented with "custom code here" that users can use to do customized stuff. Other parts should remain unchanged.

``` r
custom_input_fn <-  function(
  x,
  features,
  response = NULL,
  features_as_named_list = TRUE)
{
  validate_input_fn_args(x, features, response, features_as_named_list)
  fn <- function() {
    if (features_as_named_list) {
      # For canned estimators
      input_features <- lapply(features, function(feature) {
        custom_function(as.character(x[[feature]])) # custom code here
      })
      names(input_features) <- features
    } else {
      # For custom estimators
      input_features <- custom_function(as.matrix(x[, features]), ...) # custom code here
    }
    if (!is.null(response)) {
      input_response <- custom_function(x[[response]], ...) # custom code here
    } else {
      input_response <- NULL
    }
    list(input_features, input_response)
  }
  return(list(
    input_fn = fn,
    features_as_named_list = features_as_named_list))
}

```

## Feature Columns Layers

The feature column layers API are basically wrappers around `tf.contrib.layers.feature_column`, for example, `column_real_valued` is `tf.contrib.layers.feature_column.real_valued_column`, we wrap it this way so users can just type `column_` and utilize the autocomplete functionality in RStudio as well as reducing the appearances of `$` in the code. These are used together with spec constructors. Right now all arguments are just `...`, which means that users will need to look up Python API documentation themselves. A more general and automatic way of generating these wrapper APIs is needed.


## Custom Estimator

The following code snippet demonstrates the construction and fitting of a custom estimator that has custom architectures. Users define the model architecture in a custom model function `custom_model_fn` that contains the following arguments in the signature that users can grab to define customized handling conditionally:

* features and labels of the model.
* mode that contains the different modes of a model, such as training, inference, or evaluation.
* params that contains the tuning parameters in a model.
* config that represents the `RunConfig` objects used in a model, including GPU percentages, cluster information, etc.

The `custom_model_fn` function should return a `estimator_spec(predictions, loss, train_op, mode)` that contains the predictions, losses, training op, and mode.


``` r
constructed_input_fn <- input_fn(
	x = iris,
	response = "Species",
	features = c(
	  "Sepal.Length",
	  "Sepal.Width",
	  "Petal.Length",
	  "Petal.Width"),
	features_as_named_list = FALSE
)

custom_model_fn <- function(features, labels, mode, params, config) {
	labels <- tf$one_hot(labels, 3L)

	# Create three fully connected layers respectively of size 10, 20, and 10 with
	# each layer having a dropout probability of 0.1.
	logits <- features %>%
	  tf$contrib$layers$stack(
	    tf$contrib$layers$fully_connected, c(10L, 20L, 10L),
	    normalizer_fn = tf$contrib$layers$dropout,
	    normalizer_params = list(keep_prob = 0.9)) %>%
	  tf$contrib$layers$fully_connected(3L, activation_fn = NULL) # Compute logits (1 per class) and compute loss.

	loss <- tf$losses$softmax_cross_entropy(labels, logits)
	predictions <- list(
	  class = tf$argmax(logits, 1L),
	  prob = tf$nn$softmax(logits))

	# Create a tensor for training op.
	train_op <- tf$contrib$layers$optimize_loss(
	  loss,
	  tf$contrib$framework$get_global_step(),
	  optimizer = 'Adagrad',
	  learning_rate = 0.1)

	return(estimator_spec(predictions, loss, train_op, mode))
}

# Initialize and fit the model using the the custom model function we defined
# and the constructed_input_fn that represents the input data source.  
classifier <- estimator(
  model_fn = custom_model_fn) %>%
fit(input_fn = constructed_input_fn, steps = 2L)
```

Note that the above code contains a lot of `$`s. It is unnecessary to create wrapper APIs for every methods that users might use, e.g. `tf$contrib$layers$optimize_loss`, since custom models are designed to be flexible and extensible so users can insert any arbitrary low level TensorFlow APIs.

Users can use `coef()` to extract the trained coefficients of a model. 

``` r
coefs <- coef(classifier)
```

## Canned Estimators

For canned estimators, users need to specify the input_fn, feature columns, and other required arguments for a particular canned estimator. Note that in the following example, `linear_dnn_combined_classifier` takes two types of feature columns that are used for linear and dnn separately. 

``` r
mtcars$vs <- as.factor(mtcars$vs)
dnn_feature_columns <- feature_columns(mtcars, "drat")
linear_feature_columns <- feature_columns(mtcars, "cyl")
constructed_input_fn <- input_fn(mtcars, response = "vs", features = c("drat", "cyl"))

classifier <-
	linear_dnn_combined_classifier(
	  linear_feature_columns = linear_feature_columns,
	  dnn_feature_columns = dnn_feature_columns,
	  dnn_hidden_units = c(3L, 3L),
	  dnn_optimizer = "Adagrad"
	) %>% fit(input_fn = constructed_input_fn, steps = 2L)
```

Users can use `coef()` to extract the trained coefficients of a model.

``` r
coefs <- coef(classifier)
```

Once a model is trained, users can use `predict()` that makes predictions on a given input_fn that represents the inference data source. an argument named `type` can be `"raw"` so `predict()` will return the raw predictions outcomes, as well as `"prob"` and `"logistic"` that returns prediction probabilities and logistics if a model is of classification type.

``` r
predictions <- predict(classifier, input_fn = constructed_input_fn)
predictions <- predict(classifier, input_fn = constructed_input_fn, type = "prob")
predictions <- predict(classifier, input_fn = constructed_input_fn, type = "logistic")
```

## Run Options

All estimators accept an argument called `run_options` that is a `run_options` object containing the `model_dir` and `RunConfig` that specifies the checkpoint directory and the model run-time configuration, such as cluster information, GPU fractions, etc. If not specified, default values will be used.


## Experiments

Experiments are designed for easier experiments, e.g. define your model, specify training and evaluation data and steps, frequencies, where to run, metrics to use to monitor the process, etc. They contain all neccessary information required to run experiments and can be easily packed up to run in places like CloudML, local environment, or cluster.

``` r
clf <-
  linear_dnn_combined_classifier(
    linear_feature_columns = linear_feature_columns,
    dnn_feature_columns = dnn_feature_columns,
    dnn_hidden_units = c(3L, 3L),
    dnn_optimizer = "Adagrad"
  ) %>% fit(input_fn = input_fn)

experiment <- experiment(
  clf,
  train_input_fn = input_fn,
  eval_input_fn = input_fn,
  train_steps = 3L,
  eval_steps = 3L,
  continuous_eval_throttle_secs = 60L
)

experiment_result <- train_and_evaluate(experiment)

```

