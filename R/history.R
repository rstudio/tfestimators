new_tf_estimator_history <- function(losses = NULL, step = NULL) {
  metrics <- names(losses)
  steps <- tail(step, 1)
  structure(
    list(
      params = list(metrics = metrics,
                    steps = steps),
      losses = losses, 
      step = step
    ),
    class = "tf_estimator_history"
  )
}

#' @export
as.data.frame.tf_estimator_history <- function(x, ...) {
  df <- data.frame(x[["losses"]]) %>%
    cbind(data.frame(x["step"]))
  if (length(df))
    tidyr::gather(df, "metric", "value", -"step")
  else df
}

#' @export
print.tf_estimator_history <- function(x, ...) {
  print(as.data.frame(x), ...)
}

#' Plot training history
#' 
#' Plots metrics recorded during training. 
#' 
#' @param x Training history object returned from `train()`.
#' @param y Unused.
#' @param metrics One or more metrics to plot (e.g. `c('total_losses', 'mean_losses')`).
#'   Defaults to plotting all captured metrics.
#' @param method Method to use for plotting. The default "auto" will use 
#'   \pkg{ggplot2} if available, and otherwise will use base graphics.
#' @param smooth Whether a loess smooth should be added to the plot, only 
#'   available for the `ggplot2` method. If the number of data points is smaller
#'   than ten, it is forced to false.
#' @param theme_bw Use `ggplot2::theme_bw()` to plot the history in 
#'   black and white.
#' @param ... Additional parameters to pass to the [plot()] method.
#' 
#' @export
plot.tf_estimator_history <- function(x, y, metrics = NULL, method = c("auto", "ggplot2", "base"),
                                      smooth = getOption("tf.estimator.plot.history.smooth", TRUE),
                                      theme_bw = getOption("tf.estimator.plot.history.theme_bw", FALSE),
                                      ...) {
  # check which method we should use
  method <- match.arg(method)
  if (method == "auto") {
    if (requireNamespace("ggplot2", quietly = TRUE))
      method <- "ggplot2"
    else
      method <- "base"
  }
  
  # convert to data frame
  df <- x %>%
    compose_history_metadata(rename_step_col = FALSE) %>%
    tidyr::gather("metric", "value", -"step")
  
  # if metrics is null we plot all of the metrics
  if (is.null(metrics)) metrics <- x$params$metrics
  
  # select the correct metrics
  df <- df[df$metric %in% metrics, ]
  
  if (method == "ggplot2") {
    # helper function for correct breaks (integers only)
    int_breaks <- function(x) pretty(x)[pretty(x) %% 1 == 0]
    
    p <- ggplot2::ggplot(df, ggplot2::aes_(~step, ~value))
    
    smooth_args <- list(se = FALSE, method = 'loess', na.rm = TRUE)
    
    if (theme_bw) {
      smooth_args$size <- 0.5
      smooth_args$color <- "gray47"
      p <- p +
        ggplot2::theme_bw() +
        ggplot2::geom_point(col = 1, na.rm = TRUE, size = 2) +
        ggplot2::scale_shape(solid = FALSE)
    } else {
      p <- p +
        ggplot2::geom_point(shape = 21, col = 1, na.rm = TRUE)
    }
    
    if (smooth && nrow(df) >= 10)
      p <- p + do.call(ggplot2::geom_smooth, smooth_args)
    
    p <- p +
      ggplot2::facet_grid(metric~., switch = 'y', scales = 'free_y') +
      ggplot2::scale_x_continuous(breaks = int_breaks) +
      ggplot2::theme(axis.title.y = ggplot2::element_blank(), strip.placement = 'outside',
                     strip.text = ggplot2::element_text(colour = 'black', size = 11),
                     strip.background = ggplot2::element_rect(fill = NA, color = NA))
    
    return(p)
  }
  
  if (method == 'base') {
    # par
    op <- par(mfrow = c(length(metrics), 1),
              mar = c(3, 3, 2, 2)) # (bottom, left, top, right)
    on.exit(par(op), add = TRUE)
    
    for (i in seq_along(metrics)) {
      
      # get metric
      metric <- metrics[[i]]
      
      # adjust margins
      top_plot <- i == 1
      bottom_plot <- i == length(metrics)
      if (top_plot)
        par(mar = c(1.5, 3, 1.5, 1.5))
      else if (bottom_plot)
        par(mar = c(2.5, 3, .5, 1.5))
      else
        par(mar = c(1.5, 3, .5, 1.5))
      
      # select data for current panel
      df2 <- df[df$metric == metric, ]
      
      # plot values
      plot(df2$step, df2$value, pch = c(1, 4)[df2$data],
           xaxt = ifelse(bottom_plot, 's', 'n'), xlab = "step", ylab = metric, ...)
      
      # add legend
      legend_location <- ifelse(
        df2[,'value'][1] > df2[,'value'][x$params$steps],
        "topright", "bottomright")
      graphics::legend(legend_location, legend = metric, pch = 1)
    }
  }
}

compose_history_metadata <- function(history, max_rows = 100, rename_step_col = TRUE) {
  training_history <- as.data.frame(history) %>%
    tidyr::spread("metric", "value")
  
  training_history <- if (nrow(training_history) > max_rows) {
    # cap number of points plotted
    nrow_history <- nrow(training_history)
    sampling_indices <- seq(1, nrow_history, by = nrow_history / max_rows) %>%
      as.integer()
    training_history[sampling_indices,]
  } else training_history
  
  if (rename_step_col)
    names(training_history)[names(training_history) == "step"] <- "epoch"
  training_history
}
