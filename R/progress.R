hook_progress_bar <- function(steps) {
  
  format <- if (is.null(steps))
    "[:spin] - loss: :loss"
  else
    ":current/:total [:bar] - ETA: :eta - loss: :loss"
  
  bar <- progress::progress_bar$new(
    format = format,
    total = steps,
    complete = "=",
    incomplete = ".",
    clear = FALSE
  )
  
  session_run_hook(
    
    before_run = function(context) {
      session_run_args(run_context_losses(context))
    },
    
    after_run = function(context, values) {
      loss <- values$results[[length(values$results)]]
      tokens <- list(loss = format(round(loss, 2), nsmall = 2))
      bar$tick(tokens = tokens)
    }
    
  )
}
