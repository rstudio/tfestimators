# High-level Architecture Notes 

## Formula Interface

We have basic formula support, e.g. `simple_custom_model_recipe(y ~ x1 + x2)`, for defining targets vs. features but we are investigating how to achieve something like this `y ~ tf$contrib$layers$real_valued_column(x, someAddtionalArgs) + x2`. We may decide not to support this at all since it's not very customizable. Alternatively, we can provide better helper functions for users to achieve the same thing.

## Recipes

A recipe includes `input_fn` and `feature_columns` to define the input as well as features vs. targets. Users can use default recipes which just convert numeric columns into `real_valued_column`, factor columns into `sparse_column_with_keys`, etc. Users can also define their own recipes using rather low level APIs. We will provide more helper functions for users to construct customized recipes. Note that `feature_columns` are only useful for linear and DNN models. For other more advanced deep learning models, people usually just focus on defining the model architecture that could serve as additional features.

For examples, in order to define a custom estimator, we have the following three approaches:

``` r
# Formula interface
formula_recipe <- simple_custom_model_recipe(
  Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
  data = iris,
  model_fn = custom_model_fn)
  
# Simple non-formula interface
simple_recipe <- simple_custom_model_recipe(
  x = iris,
  response = "Species",
  features = c(
    "Sepal.Length",
    "Sepal.Width",
    "Petal.Length",
    "Petal.Width"),
  model_fn = custom_model_fn)
  
# Custom interface (highly customizable)
recipe <- custom_model_recipe(model_fn = custom_model_fn, input_fn = iris_input_fn)
```

## Run Options

Provides run-time configurations/options for model initialization and fitting, e.g. `steps`, `RunConfig`, `model_dir`, etc. We can define this `run_options` and pass it to built-in estimators or custom estimators.

```
r_opts = run_options(steps = 2, run_config = learn$RunConfig(tf_random_seed = 1))
classifier <- create_custom_estimator(custom_model_fn, iris_input_fn, run_options = r_opts)
```

## Linear Models

```
recipe <- simple_linear_recipe(mtcars, "mpg", "drat")
reg <- linear_regression(recipe = recipe)

# Generate predictions. Note that type = 'prob' is only available for classification model
predictions <- predict(reg, newdata = mtcars, type = 'raw')

# Obtain the coefficients
coefs <- coef(reg)
```

## Linear and DNN Combined Models

```
recipe <-
simple_linear_dnn_combined_recipe(
  mtcars,
  response = "mpg",
  linear_features = c("cyl"),
  dnn_features = c("drat")
)

reg <-
linear_dnn_combined_regression(
  recipe = recipe,
  dnn_hidden_units = c(1L, 1L),
  dnn_optimizer = "Adagrad"
)
# predict(), coef() is similar to linear models
```

## Custom Models
```
iris_data <- learn$datasets$load_dataset("iris")

custom_model_fn <- function(features, target) {
	target <- tf$one_hot(target, 3L)
    # Create three fully connected layers respectively of size 10, 20, and 10 with
    # each layer having a dropout probability of 0.1.
    logits <- features %>%
      tf$contrib$layers$stack(
        tf$contrib$layers$fully_connected, c(10L, 20L, 10L),
        normalizer_fn = tf$contrib$layers$dropout,
        normalizer_params = list(keep_prob = 0.9)) %>%
      tf$contrib$layers$fully_connected(3L, activation_fn = NULL) # Compute logits (1 per class) and compute loss.

    loss <- tf$losses$softmax_cross_entropy(target, logits)

    # Create a tensor for training op.
    train_op <- tf$contrib$layers$optimize_loss(
      loss,
      tf$contrib$framework$get_global_step(),
      optimizer = 'Adagrad',
      learning_rate = 0.1)

    return(custom_model_return_fn(logits, loss, train_op))
}
```

Note that above we have a wrapper for the return signature using custom_model_return_fn(). This is to avoid as much manual mistake as possible by the users. It also converts the logits so later users can call predict() for both raw predictions and probabilities.

```

recipe <- custom_model_recipe(model_fn = custom_model_fn, input_fn = iris_input_fn)
classifier <- create_custom_estimator(recipe)

predictions <- predict(classifier, input_fn = iris_input_fn, type = "raw")
```


## Experiments

Experiments are designed for easier experiments, e.g. define your model, specify training and evaluation data and steps, frequencies, where to run, etc. 

```
clf <- linear_dnn_combined_classification(
	recipe = recipe,
	dnn_hidden_units = c(1L, 1L),
	dnn_optimizer = "Adagrad",
	skip_fit = TRUE
)

experiment <- setup_experiment(
	clf,
	train_data = mtcars,
	eval_data = mtcars,
	train_steps = 3L,
	eval_steps = 3L,
	continuous_eval_throttle_secs = 60L
)

experiment_result <- experiment$train_and_evaluate()
```
