context("Testing examples")

source("helper-utils.R")

# some helpers
run_example <- function(example_path) {
  env <- new.env()
  capture.output({
    example_path <- system.file("examples", example_path, package = "tfestimators")
    old_wd <- setwd(dirname(example_path))
    on.exit(setwd(old_wd), add = TRUE)
    source(basename(example_path), local = env)
  }, type = "output")

  rm(list = ls(env), envir = env)
  gc()
}

examples <- NULL
# examples <- if (TRUE) {
#   c(
#     "tensorflow_layers.R",
#     "custom_estimator.R",
#     "iris_custom_decay_dnn.R"
#   )
# }

if (!is.null(examples)) {
  for (example in examples) {
    test_succeeds(paste(example, "example runs successfully"), {
      skip_if_no_tensorflow()
      expect_error(run_example(example), NA)
    })
  }
}
