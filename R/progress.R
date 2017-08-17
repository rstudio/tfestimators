hook_progress_bar <- function(label, steps) {
  
  format <- if (is.null(steps))
    paste("[:spin]", label, "-- loss: :loss, step: :step")
  else
    ":current/:total [:bar] - ETA: :eta - loss: :loss"
  
  .values <- NULL
  .n <- 0
  .bar <- progress::progress_bar$new(
    format = format,
    total = steps %||% 1E6,
    complete = "=",
    incomplete = ".",
    clear = FALSE,
    width = min(getOption("width"), 80),
    stream = stdout(),
    show_after = 0
  )
  
  session_run_hook(
    
    before_run = function(context) {
      session_run_args(run_context_losses(context))
    },
    
    after_run = function(context, values) {
      # update progress bar
      loss <- values$results[[length(values$results)]]
      tokens <- list(loss = format(round(loss, 2), nsmall = 2), step = .n + 1)
      .bar$tick(tokens = tokens)
      
      # save and update state
      .values <<- values
      .n <<- .n + 1
    },
    
    end = function(session) {
      
      # if we ran as many steps as expected, bail
      if (identical(.n, steps))
        return()
      
      # otherwise, write a single-tick progress bar encoding the finished state
      .bar <<- progress::progress_bar$new(
        format = format,
        total = .n,
        complete = "=",
        incomplete = ".",
        clear = FALSE,
        width = min(getOption("width"), 80),
        stream = stdout(),
        show_after = 0
      )
      
      # update progress bar
      values <- .values
      loss <- values$results[[length(values$results)]]
      tokens <- list(loss = format(round(loss, 2), nsmall = 2), step = .n + 1)
      .bar$tick(len = .n, tokens = tokens)
    }
    
  )
}
