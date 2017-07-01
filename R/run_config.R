#' Run Configuration
#' 
#' This class specifies the configurations for an `Estimator` run.
#' 
#' @examples
#' config <- run_config()
#' 
#' # Get the properties of the config
#' names(config)
#' 
#' # Change the mutable properties of the config
#' config <- config$replace(tf_random_seed = 11L, save_summary_steps = 12L)
#' 
#' # Print config as key value pairs
#' print(config)
#' 
#' @family run_config methods
#'   
#' @export
run_config <- function() {
  estimator_lib$run_config$RunConfig()
}

#' @export
print.tensorflow.python.estimator.run_config.RunConfig <- function(config) {
  config_names <- names(config)
  config_items <- unlist(lapply(config_names, function(item) {
    if (is.null(config[[item]])) {
      item_value <- "NULL"
    } else if (config[[item]] == "") {
      item_value <- '""'
    } else {
      item_value <- config[[item]]
    }
    paste(item,
          item_value,
          collapse = "", sep = " = ")
  }))
  cat(paste(config_items, collapse = ", "))
}


#' Task Types
#' 
#' This constant class gives the constant strings for available task types
#' used in `run_config`.
#' 
#' @examples
#' task_type()$MASTER
#' 
#' @export
#' @family run_config methods
task_type <- function() {
  estimator_lib$run_config$TaskType
}
