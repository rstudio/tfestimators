context("Test variable name and value extraction")

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

test_that("get_variable_names() error on untrained model", {
  expect_error(get_variable_names(model),
               "'get_variable_names' must be called on a trained model")
})

test_that("get_variable_value() error on untrained model", {
  expect_error(get_variable_value(model),
               "'get_variable_value' must be called on a trained model")
})

model %>% train(mtcars_input_fn(mtcars))

test_that("get_variable_names() works properly", {
  expect_identical(get_variable_names(model)[[1]], "global_step")
})

test_that("get_variable_value() works properly", {
  expect_identical(get_variable_value(model, "global_step"), 
                   list(global_step = 1))
  expect_identical(get_variable_value(model) %>%
                     names(),
                   get_variable_names(model))
})

test_that("get_variable_value() errors when variable isn't found", {
  expect_error(get_variable_value(model, "foo"),
               "Variable not found: foo")
})
