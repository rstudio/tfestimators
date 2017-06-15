context("Testing feature columns")

test_that("feature columns can be constructed correctly", {
  
  fcs <- feature_columns(mtcars, "drat")
  expect_equal(length(fcs), 1)
  expect_true(grepl("NumericColumn", class(fcs[[1]])[1]))
  fcs <- feature_columns(mtcars, c("drat", "cyl"))
  expect_equal(length(fcs), 2)
  expect_true(grepl("NumericColumn", class(fcs[[1]])[1]))
})
