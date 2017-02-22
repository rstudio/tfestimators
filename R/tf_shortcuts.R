.shortcuts <- new.env(parent = emptyenv())

setup_shortcuts <- function(env = .shortcuts) {
  
  # TODO: These need to be updated after upgrading TF
  shortcuts <- list(
    learn_lib              = tf$contrib$learn,
    learn_models_lib       = tf$contrib$learn$models,
    learn_datasets_lib     = tf$contrib$learn$datasets,
    contrib_layers_lib     = tf$contrib$layers,
    contrib_losses_lib     = tf$contrib$losses,
    contrib_variables      = tf$contrib$framework$python$ops$variables,
    feature_column_lib     = tf$contrib$layers$feature_column
  )
  
  list2env(shortcuts, envir = .shortcuts)
  
}
