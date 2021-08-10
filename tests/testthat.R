

library(testthat)
library(tfestimators)

if (identical(Sys.getenv("NOT_CRAN"), "true")) 
  test_check("tfestimators")

