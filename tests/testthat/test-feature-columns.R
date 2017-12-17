context("Testing feature columns")

source("helper-utils.R")

test_succeeds("feature columns can be constructed correctly", {
  
  fcs <- feature_columns(column_numeric("drat"))
  expect_equal(length(fcs), 1)
  expect_true(grepl("NumericColumn", class(fcs[[1]])[1]))
  fcs <- feature_columns(column_numeric("drat", "cyl"))
  expect_equal(length(fcs), 2)
  expect_true(grepl("NumericColumn", class(fcs[[1]])[1]))
})

test_succeeds("feature columns can be constructed with (cond) ~ (op) syntax", {
  
  names <- do.call(paste0, expand.grid(letters, 0:9))
  
  # bare calls (no extra arguments to column function)
  mild <- feature_columns(
    column_numeric(starts_with("a")),
    names = names
  )
  
  spicy <- feature_columns(
    starts_with("a") ~ column_numeric(),
    names = names
  )
  
  expect_equal(mild, spicy)
  
  # extra arguments to 'column_numeric()'
  mild <- feature_columns(
    column_numeric(starts_with("a"), shape = 1L),
    names = names
  )
  
  spicy <- feature_columns(
    starts_with("a") ~ column_numeric(shape = 1L),
    names = names
  )
  
  expect_equal(mild, spicy)
  
})

test_succeeds("duplicates columns are dropped", {
  
  names <- c("aa")
  columns <- feature_columns(
    column_numeric(starts_with("a")),
    column_numeric(ends_with("a")),
    names = names
  )
  
  expect_true(length(columns) == 1)
})
