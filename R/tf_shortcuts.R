.shortcuts <- new.env(parent = emptyenv())

setup_shortcuts <- function(env = .shortcuts) {
  
  # TODO: These need to be updated after upgrading TF
  shortcuts <- list(
    learn_lib              = tf$contrib$learn,
    learn_models_lib       = tf$contrib$learn$models,
    learn_datasets_lib     = tf$contrib$learn$datasets,
    contrib_layers_lib     = tf$contrib$layers,
    contrib_losses_lib     = tf$contrib$losses,
    contrib_optimizers_lib = tf$contrib$layers$optimizers,
    contrib_variables      = tf$contrib$framework$python$ops$variables,
    estimators_lib         = tf$contrib$learn$estimators,
    run_config_lib         = tf$contrib$learn$estimators$run_config,
    experiment_lib         = tf$contrib$learn$Experiment,
    feature_column_lib     = tf$contrib$layers$feature_column,
    feature_column_ops_lib = tf$contrib$layers$feature_column_ops
  )
  
  list2env(shortcuts, envir = .shortcuts)
  
}
