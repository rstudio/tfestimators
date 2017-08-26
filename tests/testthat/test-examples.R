context("Testing examples")

skip_if_no_tensorflow <- function() {
  if (is.null(tensorflow::tf))
    skip("TensorFlow not available for test")
}


# some helpers
run_example <- function(example_path) {
  env <- new.env()
  source(example_path, local = env)
  rm(list = ls(env), envir = env)
  gc()
}

examples <- if (TRUE) {
  vignettes_examples_dir <- "../../vignettes/examples"
  c(
    file.path(vignettes_examples_dir, "tensorflow_layers.R"),
    file.path(vignettes_examples_dir, "custom_estimator.R")
    )
}

if (!is.null(examples)) {
  for (example in examples) {
    test_that(paste(example, "example runs successfully"), {
      skip_if_no_tensorflow()
      expect_error(run_example(example), NA)
    })
  }
}
