# Tools for introspecting the run context of a TensorFlow session.
run_context_losses <- function(context) {
  session <- context$session
  graph <- session$graph
  session$run(graph$get_collection("losses"))[[1]]
}

run_context_global_step <- function(context) {
  session <- context$session
  graph <- session$graph
  session$run(graph$get_collection("global_step"))[[1]]
}
