#' Run Configuration
#' 
#' This class specifies the configurations for an `Estimator` run.
#' 
#' @examples
#' \dontrun{
#' config <- run_config()
#' # Get the properties of the config
#' config$keep_checkpoint_every_n_hours
#' 
#' # Change the mutable properties of the config
#' config$replace(tf_random_seed = 11, save_summary_steps = 12)
#' }
#' 
#' @family run_config methods
#'   
#' @export
run_config <- function() {
  estimator_lib$run_config$RunConfig()
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
