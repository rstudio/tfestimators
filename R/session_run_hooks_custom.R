should_execute <- function(current_step, every_n_step) {
  (current_step %% every_n_step == 0) || (current_step == 1)
}

#' A Custom Run Hook for Saving Metrics History
#' 
#' This hook allows users to save the metrics history produced during training or evaluation in
#' a specified frequency.
#' 
#' @param every_n_step Save the metrics every N steps
#' 
#' @family session_run_hook wrappers
#' @export
#' 
hook_history_saver <- function(every_n_step = 10) {
  
  hook_fn <- function(mode_key) {

    .iter_count <- 0

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
          .globals$history[[mode_key]]$step <- unlist(c(.globals$history[[mode_key]]$step, global_step))
        }
      }
    )
  }
  
  list(hook_fn = hook_fn, type = "hook_history_saver")
}

#' A Custom Run Hook to Create and Update Progress Bar During Training or Evaluation
#' 
#' This hook creates a progress bar that creates and updates the progress bar during training
#' or evaluation. 
#' 
#' @family session_run_hook wrappers
#'
#' @export
hook_progress_bar <- function() {
  
  hook_fn <- function(label, steps) {
    format <- if (is.null(steps))
      paste("[:spin]", label, "-- loss: :loss, step: :step")
    else
      paste(label, ":current/:total [:bar] - ETA: :eta - loss: :loss")
    
    .values <- NULL
    .n <- 0L
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
        .n <<- .n + 1L
      },
      
      end = function(session) {
        # notify user if didn't train for steps specified
        if (!is.null(steps) && !identical(.n, steps)) {
          op <- switch(label,
                       Training = "Training",
                       Evaluating = "Evaluation",
                       label)
          msg <- paste0("\n", op, " completed after ", .n, " steps ",
                        "but ", steps, " steps was specified")
          message(msg)
        }
      }
      
    )
  }
  
  list(hook_fn = hook_fn, type = "hook_progress_bar")
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
