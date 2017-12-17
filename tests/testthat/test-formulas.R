context("Testing formulas")

source("helper-utils.R")

test_succeeds("parse_formula parses formula correct", {
  parsed <- parse_formula(y ~ tf$Tensor(x))
  expect_equal(parsed$features, "tf$Tensor(x)")
  expect_equal(parsed$response, "y")
  expect_equal(parsed$intercept, TRUE)
  
  parsed <- parse_formula(y ~ tf$Tensor(x) + x1)
  expect_equal(parsed$features, c("tf$Tensor(x)", "x1"))
})
