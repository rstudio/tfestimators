should_execute <- function(current_step, every_n_step) {
  current_step %% every_n_step == 0
}

hook_history_saver <- function(every_n_step = 2) {
  
  hook_fn <- function(mode_key) {

    .iter_count <<- 0

    session_run_hook(
      
      before_run = function(context) {
        session_run_args(
          global_step = run_context_global_step(context),
          losses = run_context_losses(context)
        )
      },
      
      after_run = function(context, values) {
        .iter_count <<- .iter_count + 1
        if (should_execute(.iter_count, every_n_step)) {
          
          results <- values$results
          raw_losses <- results$losses[[1]]
          global_step <- results$global_step
          
          .globals$history[[mode_key]]$losses$mean_losses <- c(.globals$history[[mode_key]]$losses$mean_losses, mean(raw_losses))
          .globals$history[[mode_key]]$losses$total_losses <- c(.globals$history[[mode_key]]$losses$total_losses, sum(raw_losses))
          .globals$history[[mode_key]]$steps <- unlist(c(.globals$history[[mode_key]]$steps, global_step))
        }
      }
    )
  }
  
  list(hook_fn = hook_fn, type = "hook_history_saver")
}

# NOTE: we need to pad with an extra row of data to signal to
# the viewer that there is more data incoming. by returning a
# metrics dataframe with no padding, we signal to the viewer
# that there is no more data incoming
get_metrics_df <- function(mode_key, finalize = TRUE, steps = NULL) {
  df <- as.data.frame(.globals$history[[mode_key]]$losses)
  if (finalize)
    return(df)
  pad(df, steps %||% nrow(df) + 1)
}

#' Visualize the training or evaluation metrics
#' 
#' @param mode_key The mode when the metrics were collected that you want to visualize, e.g. "train"
#' 
#' @return The directory used to save the metrics metadata and generated html file.
visualize_metrics <- function(mode_key) {
  if (!mode_key %in% c(mode_keys()$TRAIN, mode_keys()$EVAL)) {
    stop("Only training and evaluation metrics can be visualized.")
  }
  metrics_df <- get_metrics_df(mode_key)
  if (all(dim(metrics_df) == c(0, 0))) {
    stop("No metrics available yet to be visualized in the mode you provided.")
  }
  tfruns::view_run_metrics(metrics_df)
}

# TODO: This is currently broken
hook_view_metrics <- function(every_n_step = 2) {
  
  hook_fn <- function(props, mode_key) {

    steps <- props$steps
    .metrics_viewer <- NULL
    .time <- Sys.time() - 1.0 # forces immediate update
    .iter_count <<- 0
    
    on_metrics <- function(finalize = FALSE) {
      
      # update and record metrics
      metrics_df <- get_metrics_df(mode_key, finalize, steps)

      if (is.null(.metrics_viewer)) {
        .metrics_viewer <<- tfruns::view_run_metrics(metrics_df)
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
    
    write_run_properties <- function(props) {
      properties <- list()
      properties$steps <- steps
      properties$model <- props$model
      tfruns::write_run_metadata("properties", properties)
    }
    
    session_run_hook(
      before_run = function(context) write_run_properties(props),
      after_run = function(context, values) {
        .iter_count <<- .iter_count + 1
        if (should_execute(.iter_count, every_n_step)) {
          on_metrics(FALSE)
        }
      },
      end = function(session) {
        if (should_execute(.iter_count, every_n_step)) {
          on_metrics(TRUE) 
        }
      }
    )
  }
  
  list(hook_fn = hook_fn, type = "hook_view_metrics")
}

hook_progress_bar <- function() {
  
  hook_fn <- function(label, steps) {
    format <- if (is.null(steps))
      paste("[:spin]", label, "-- loss: :loss, step: :step")
    else
      paste(label, ":current/:total [:bar] - ETA: :eta - loss: :loss")
    
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
  
  list(hook_fn = hook_fn, type = "hook_progress_bar")
}
