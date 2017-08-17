# Tools for introspecting the run context of a TensorFlow session.
run_context_losses <- function(context) {
  context$session$graph$get_collection("losses")
}

run_context_global_step <- function(context) {
  context$session$graph$get_collection("global_step")
}
