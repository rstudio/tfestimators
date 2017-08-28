hook_history_saver <- function() {
  session_run_hook(
    after_run = function(run_context, run_values) {
      session <- run_context$session
      graph <- session$graph
      global_step_op <- graph$get_collection(graph_keys()$GLOBAL_STEP)
      losses_op <- graph$get_collection(graph_keys()$LOSSES)
      raw_losses <- session$run(losses_op)[[1]]
      global_step <- session$run(global_step_op)
      .globals$history$losses$mean_losses <- c(.globals$history$losses$mean_losses, mean(raw_losses))
      .globals$history$losses$total_losses <- c(.globals$history$losses$total_losses, sum(raw_losses))
      .globals$history$steps <- unlist(c(.globals$history$steps, global_step))
    }
  )
}

hook_view_metrics <- function(steps) {
  
  force(steps)
  
  .metrics_viewer <- NULL
  .time <- Sys.time()
  
  get_metrics_df <- function() {
    df <- as.data.frame(.globals$history$losses)
    if (!is.null(steps)) {
      df <- pad(df, steps)
    }
    df
  }
  
  on_metrics <- function() {
    
    # update and record metrics
    metrics_df <- get_metrics_df()
    if (is.null(.metrics_viewer)) {
      .metrics_viewer <<- tfruns::view_run_metrics(metrics_df)
      Sys.sleep(0.1)
    } else {
      tfruns::update_run_metrics(.metrics_viewer, metrics_df)
    }
    
    # record metrics
    tfruns::write_run_metadata("metrics", metrics_df)
    
    # pump events (once every second)
    now <- Sys.time()
    if (now - .time > 1.0) {
      .time <<- now
      Sys.sleep(0.1)
    }
    
  }
  
  session_run_hook(
    after_run = function(context, values) on_metrics()
  )
}

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
  
  update_progress <- function(values, n = 1) {
    losses <- values$results[["losses"]]
    loss <- losses[[length(losses)]]
    tokens <- list(loss = format(round(loss, 2), nsmall = 2), step = .n + 1)
    .bar$tick(len = n, tokens = tokens)
  }
  
  session_run_hook(
    
    before_run = function(context) {
      session_run_args(
        losses = run_context_losses(context)
      )
    },
    
    after_run = function(context, values) {
      
      # update progress bar
      update_progress(values)
      
      # save and update state
      .values <<- values
      .n <<- .n + 1
    },
    
    end = function(session) {
      
      # if we ran as many steps as expected, bail
      if (is.null(steps) || identical(.n, steps))
        return()

      update_progress(.values, steps - .n)
    }
    
  )
}
