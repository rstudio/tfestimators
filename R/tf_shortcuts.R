#' @export
setup_shortcuts <- function(env = .GlobalEnv) {
  assign("learn_lib", tf$contrib$learn, envir = env)
  assign("contrib_layers_lib", tf$contrib$layers, envir = env)
  assign("contrib_losses_lib", tf$contrib$losses, envir = env)
  assign("contrib_optimizers_lib", contrib_layers_lib$optimizers, envir = env)
  assign("contrib_variables", tf$contrib$framework$python$ops$variables, envir = env)
  assign("estimators_lib", learn_lib$estimators, envir = env)
  assign("run_config_lib", estimators_lib$run_config, envir = env)
  assign("experiment", learn_lib$Experiment, envir = env)
  assign("feature_column_lib", tf$contrib$layers$feature_column, envir = env)
  assign("feature_column_ops_lib", tf$contrib$layers$feature_column_ops, envir = env)
  assign("learn_models_lib", learn_lib$models, envir = env)
  assign("learn_datasets_lib", learn_lib$datasets, envir = env)
}
