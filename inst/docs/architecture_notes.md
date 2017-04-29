# Architecture and Main Components

## Spec Constructors

Spec constructors are for constructing the input and features for a particular estimator. All estimators require input function. Canned estimators, in addition to the requirement of input function, require specification for feature columns. 

### Feature Columns

The first spec constructor is feature columns required for canned estimators such as `DNNEstimator`. This specifies the feature transformations and combinations for a model, e.g. `column_embedding()` that converts a categorical variable into embedding and `column_crossed()` that combines two variables in a specified way. 

Users can use the default `feature_columns()` function to convert columns in an automatic fashion without any fancy feature engineering, e.g. numeric variables are converted using `column_real_valued()`, factor variables are converted using `column_with_keys()`, and character variables are converted using `column_with_hash_bucket()`. 

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

The feature columns transformation functions are wrappers around `tf.contrib.layers.feature_column`, for example, `column_real_valued()` is `tf.contrib.layers.feature_column.real_valued_column`, we wrap it this way so users can just type `column_` and utilize the autocomplete functionality in RStudio to find available types of feature columns faster as well as reducing the appearances of `$` in the code. These are used together with spec constructors that are used often for constructing canned estimator's features. A variety of feature column funcions are available. For example, `column_one_hot()` specifies a feature column that's one-hot encoded, `column_sparse_weighted()` creates a feature column in combination with a designated weight column.

### Input Function

Another spec constructor is the input function required for the estimators. This is where users provide the input sources to feed into the model, e.g. in-memory dataframe/matrix, streaming data, serialized data formats, etc. 

Users have two ways to specify in-memory data set - using formula interface or passing `features` and `response` arguments. For example, users can use built-in `input_fn()` on `data.frame` objects like the following:

``` r
input_fn(mtcars, response = "mpg", features = c("drat", "cyl"))
```

or use the formulate interface like below where left-hand and right-hand side of the `~` represent response column and feature columns respectively:

``` r
input_fn(mpg ~ drat + cyl, data = mtcars)
```

Note that there's an argument named `features_as_named_list` that should be `TRUE` if this is used by a canned estimator and should be `FALSE` if this is used by a custom estimator. 

There's also a built-in `input_fn()` that works on nested lists, for example:

``` r
input_fn(
  x = list(
    features = list(
      list(list(1), list(2), list(3)),
      list(list(4), list(5), list(6))),
    response = list(
      list(1, 2, 3), list(4, 5, 6))),
  features = "features",
  response = "response")
````

In the above example, the data is a list of two named lists where each named list can be seen as different columns in a dataset. In this case, a column named `features` is being used as features to the model and a column named `response` is being used as the response variable. This nested lists format is particularly useful when constructing sequence input to Recurrent Neural Networks (RNN). Once the data is defined using `input_fn()`, it can be used directly in RNN constructor.

Users can also write custom input function, e.g. a function `custom_input_fn()`, to convert each feature into a `Tensor` or `SparseTensor` according to the needs. This function should return a list that consists of `input_fn` and `features_as_named_list` so the custom or canned estimator can recognize them. The following skeleton code has a few places commented with "custom code here" that users can use to do customized operation. Other parts should remain unchanged.

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
        custom_function(x[[feature]], ...) # custom code here
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

Users are encounraged to follow the above skeleton but it may not be suitable for all types of models. For example, if a user want to construct some complicated input, such as a batched sequence input similar to a sine curve for feeding RNNs, he can define something similar to the following using mostly low-level TensorFlow APIs:

``` r
get_batched_sin_input_fn <- function(batch_size, sequence_length, increment, seed = NULL) {
  list(
    input_fn = function() {
      starts <- random_ops$random_uniform(
        list(batch_size), minval = 0, maxval = pi * 2.0,
        dtype = tf$python$framework$dtypes$float32, seed = seed)
      sin_curves <- functional_ops$map_fn(
        function(x){
          math_ops$sin(
            math_ops$linspace(
              array_ops$reshape(x[1], list()),
              (sequence_length - 1) * increment,
              as.integer(sequence_length + 1)))
        },
        tuple(starts),
        dtype = tf$python$framework$dtypes$float32
      )
      inputs <- array_ops$expand_dims(
        array_ops$slice(
          sin_curves,
          np$array(list(0, 0), dtype = np$int64),
          np$array(list(batch_size, sequence_length), dtype = np$int64)),
        2L
      )
      labels <- array_ops$slice(sin_curves,
                                np$array(list(0, 1), dtype = np$int64),
                                np$array(list(batch_size, sequence_length), dtype = np$int64))
      return(tuple(list(inputs = inputs), labels))
    },
    features_as_named_list = TRUE)
}
```

Users can then further define `input_fn` for training and evaluation:

``` r
train_input_fn <- get_batched_sin_input_fn(batch_size, sequence_length, pi / 32, seed = 1234)
eval_input_fn <- get_batched_sin_input_fn(batch_size, sequence_length, pi / 32, seed = 4321)
```

## Estimator

Estimator is an interface that provides an abstraction for a machine learning model. It is designed to be detailed enough to allow for downstream infrastructure to be written, but general enough to not constrain the type of model represented by an Estimator. Estimators are given input by a user defined input function, as illustrated in earlier section.

The Estimator's architecture is configured using a user-defined `model_fn`, a function which builds a TensorFlow graph and returns necessary information to train a model, evaluate it, and predict with it. Users writing custom estimators to implement custom model architecture only have to implement this function to specify the layers of the custom Estimator. It is possible, and in fact, common, that `model_fn` contains regular TensorFlow without using any other part of our framework.  It is often the case because existing models are being adapted or converted to be implemented in terms of an estimator.

This library also provides canned estimators that have already implemented the model architecture, such as linear classifier, linear regressor, DNN classifier, DNN regressor, SVM classifier, etc. Users only need to focus on the input sources and the feature columns used to train a model, evaluate it, and predict with it. Users then should be able to choose freely the level of abstraction best suited for the problem at hand.

### Custom Estimator

The following code snippet demonstrates the construction and fitting of a custom estimator that has custom architectures. Users define the model architecture in a custom model function `custom_model_fn` that contains the following arguments in the signature that users can grab to define customized handling conditionally:

* features and labels of the model.
* mode that contains the different modes of a model, such as training, inference, or evaluation.
* params that contains the tuning parameters in a model.
* config that represents the `RunConfig` objects used in a model, including GPU percentages, cluster information, etc.

The `custom_model_fn()` function should return an `estimator_spec(predictions, loss, train_op, mode)` that contains the predictions, losses, training op, and mode.


``` r
constructed_input_fn <- input_fn(
	x = iris,
	response = "Species",
	features = c(
	  "Sepal.Length",
	  "Sepal.Width",
	  "Petal.Length",
	  "Petal.Width"),
	features_as_named_list = FALSE,
	batch_size = 10L
)

custom_model_fn <- function(features, labels, mode, params, config) {
	  # Create three fully connected layers respectively of size 10, 20, and 10 with
    # each layer having a dropout probability of 0.1.
    logits <- features %>%
      tf$contrib$layers$stack(
        tf$contrib$layers$fully_connected, c(10L, 20L, 10L),
        normalizer_fn = tf$contrib$layers$dropout,
        normalizer_params = list(keep_prob = 0.9)) %>%
      tf$contrib$layers$fully_connected(3L, activation_fn = NULL) # Compute logits (1 per class) and compute loss.
    
    predictions <- list(
      class = tf$argmax(logits, 1L),
      prob = tf$nn$softmax(logits))
    
    # Return estimator_spec early with NULL loss and train_op during inference mode
    if(mode == "infer"){
      return(estimator_spec(
      predictions = predictions, mode = mode, loss = NULL, train_op = NULL))
    }
    
    labels <- tf$one_hot(labels, 3L)
    loss <- tf$losses$softmax_cross_entropy(labels, logits)
    
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
classifier <- estimator(model_fn = custom_model_fn)
classifier <- fit(classifier, input_fn = constructed_input_fn, steps = 2L)
```

Note that the above code contains a lot of `$`s. It is unnecessary to create wrapper APIs for every methods that users might use, e.g. `tf$contrib$layers$optimize_loss`, since custom models are designed to be flexible and extensible so users can insert any arbitrary low level TensorFlow APIs.

Users can then supply an `input_fn` and make predictions. 

``` r
predictions <- predict(classifier, input_fn = constructed_input_fn)
```

Since our predictions is defined as a list of two items in the custom model function like follows:

``` r
predictions <- list(
  class = tf$argmax(logits, 1L),
  prob = tf$nn$softmax(logits))
```

we can use the following trick to loop through each prediction and extract the predicted classes and probabilities.

``` r
# extract predicted classes
predicted_classes <- unlist(lapply(predictions, function(prediction) {
  prediction$class
}))

# extract predicted probabilities
predicted_probs <- lapply(predictions, function(prediction) {
  prediction$prob
})

```

### Canned Estimators

For canned estimators, users need to specify the input_fn, feature columns, and other required arguments for a particular canned estimator. Note that in the following example, `linear_dnn_combined_classifier` takes two types of feature columns that are used for linear and dnn separately. 

``` r
mtcars$vs <- as.factor(mtcars$vs)
dnn_feature_columns <- feature_columns(mtcars, "drat")
linear_feature_columns <- feature_columns(mtcars, "cyl")
custom_input_fn <- input_fn(mtcars, response = "vs", features = c("drat", "cyl"))

classifier <-
	linear_dnn_combined_classifier(
	  linear_feature_columns = linear_feature_columns,
	  dnn_feature_columns = dnn_feature_columns,
	  dnn_hidden_units = c(3L, 3L),
	  dnn_optimizer = "Adagrad"
	)

classifier <- fit(classifier, input_fn = custom_input_fn, steps = 2L)
```

Users can use `coef()` to extract the trained coefficients of a model.

``` r
coefs <- coef(classifier)
```

Once a model is trained, users can use `predict()` that makes predictions on a given input_fn that represents the inference data source. an argument named `type` can be `"raw"` so `predict()` will return the raw predictions outcomes, as well as `"prob"` and `"logistic"` that returns prediction probabilities and logistics if a model is of classification type.

``` r
predictions <- predict(classifier, input_fn = custom_input_fn)
predictions <- predict(classifier, input_fn = custom_input_fn, type = "prob")
predictions <- predict(classifier, input_fn = custom_input_fn, type = "logistic")
```

## Run Options

All estimators accept an argument called `run_options` that is a `run_options` object containing the `model_dir` and `RunConfig` that specifies the checkpoint directory and the model run-time configuration, such as cluster information, GPU fractions, etc. If not specified, default values will be used.

## SessionRunHooks

`SessionRunHooks` are useful to track training, report progress, request early stopping and more. Users can attach an arbitrary number of hooks to an estimator. `SessionRunHooks` use the observer pattern and notify at the following points:

 - when a session starts being used
 - before a call to the `session.run()`
 - after a call to the `session.run()`
 - when the session closed

A `SessionRunHook` encapsulates a piece of reusable/composable computation that can piggyback a call to `MonitoredSession.run()`. A hook can add any ops-or-tensor/feeds to the run call, and when the run call finishes with success gets the outputs it requested. Hooks are allowed to add ops to the graph in `hook.begin()`. The graph is finalized after the `begin()` method is called.

There are a few pre-defined `SessionRunHooks` available, for example:
 - `hook_stop_at_step`: Request stop based on global_step.
 - `hook_checkpoint_saver`: Saves checkpoint.
 - `hook_logging_tensor`: Outputs one or more tensor values to log.
 - `hook_nan_tensor`: Request stop if given `Tensor` contains Nans.
 - `hook_summary_saver`: Saves summaries to a summary writer.
 - `hook_global_step_waiter`: Delays execution until reaching a certain global step.

Similarly to feature columns, all available `SessionRunHooks` are named with `hook_xxx` to utilize the autocomplete functionality to speed up searching for available types of `SessionRunHooks`.

For example, in order to customize the checkpoint saving mechanism, users can initialize a monitor using `hook_checkpoint_saver()` that defines the checkpoint directory and the frequency of saving new checkpoint. 

``` r
monitor <- hook_checkpoint_saver(
  checkpoint_dir = "/tmp/ckpt_dir",
  save_secs = 2)
```

Once monitor and an estimator are defined, the monitor can be attached to the estimator via the argument `monitors` when fitting the model. 

``` r
lr <- linear_dnn_combined_regressor(
  linear_feature_columns = linear_feature_columns,
  dnn_feature_columns = dnn_feature_columns,
  dnn_hidden_units = c(1L, 1L),
  dnn_optimizer = "Adagrad"
)

lr <- fit(
  lr,
  input_fn = custom_input_fn,
  steps = 10L,
  monitors = monitor)
```

## Experiments

Experiments are designed for easier experiments, e.g. define your model, specify training and evaluation data and steps, frequencies, where to run, metrics to use to monitor the process, etc. They contain all neccessary information required, such as input_fn for both training and evaluation, to run experiments and can be easily packed up to run in places like CloudML, local environment, or cluster.

For example, we firstly construct a classifier

``` r
clf <-
  linear_dnn_combined_classifier(
    linear_feature_columns = linear_feature_columns,
    dnn_feature_columns = dnn_feature_columns,
    dnn_hidden_units = c(3L, 3L),
    dnn_optimizer = "Adagrad"
  )
  
```

and then we pass the classifier into `experiment()` together with other neccessary information, such as separate input functions for training and evaluation, training and avaluation steps, etc. Then we can call `train_and_evaluate()` to conduct the experiment by running training and evaluation altogether.

``` r

experiment <- experiment(
  clf,
  train_input_fn = custom_train_input_fn,
  eval_input_fn = custom_eval_input_fn,
  train_steps = 3L,
  eval_steps = 3L,
  continuous_eval_throttle_secs = 60L
)

experiment_result <- train_and_evaluate(experiment)

```

