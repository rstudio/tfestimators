hook_progress_bar <- function(steps) {
  
  format <- if (is.null(steps))
    "[:spin] - loss: :loss"
  else
    ":current/:total [:bar] - ETA: :eta - loss: :loss"
  
  # placeholder for progress bar
  bar <- NULL
  
  session_run_hook(
    
    begin = function() {
      
      # TODO: can we also report which epoch we're running here?
      bar <<- progress::progress_bar$new(
        format = format,
        total = steps,
        complete = "=",
        incomplete = ".",
        clear = FALSE
      )
      bar$tick(0)
      
      NULL
    },
    
    after_run = function(context, values) {
      
      # TODO: how do we detect if we've run out of items?
      tokens <- list(
        loss = format(mean(run_context_losses(context)))
      )
      
      bar$tick(tokens = tokens)
      
      NULL
    },
    
    end = function(session) {
    }
    
  )
}
