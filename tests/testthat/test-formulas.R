context("Test formulas")

test_that("parse_formula parses formula correct", {
  parsed <- parse_formula(y ~ tf$contrib$layers$real_valued_column(x))
  expect_equal(parsed$features, "tf$contrib$layers$real_valued_column(x)")
  expect_equal(parsed$response, "y")
  expect_equal(parsed$intercept, TRUE)
  
  parsed <- parse_formula(y ~ tf$contrib$layers$real_valued_column(x) + x1)
  expect_equal(parsed$features, c("tf$contrib$layers$real_valued_column(x)", "x1"))
})
