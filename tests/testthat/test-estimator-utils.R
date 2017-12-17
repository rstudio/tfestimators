context("Testing variable name and value extraction")

source("helper-utils.R")

if (have_tensorflow()) {
  mtcars_input_fn <- function(data) {
    input_fn(data, 
             features = c("disp", "cyl"), 
             response = "mpg",
             batch_size = 32)
  }
  cols <- feature_columns( 
    column_numeric("disp", "cyl")
  )
  model <- linear_regressor(feature_columns = cols)
}

test_succeeds("variable_names() error on untrained model", {
  expect_error(variable_names(model),
               "'variable_names\\(\\)' must be called on a trained model")
})

test_succeeds("variable_value() error on untrained model", {
  expect_error(variable_value(model),
               "'variable_value\\(\\)' must be called on a trained model")
})

if (have_tensorflow())
  model %>% train(mtcars_input_fn(mtcars))

test_succeeds("variable_names() works properly", {
  expect_identical(variable_names(model)[[1]], "global_step")
})

test_succeeds("variable_value() works properly", {
  expect_identical(variable_value(model, "global_step"), 
                   list(global_step = 1))
  expect_identical(variable_value(model) %>%
                     names(),
                   variable_names(model))
})

test_succeeds("variable_value() errors when variable isn't found", {
  expect_error(variable_value(model, "foo"),
               "Variable not found: foo")
})
