% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/session_run_hooks_builtin_wrappers.R
\name{hook_summary_saver}
\alias{hook_summary_saver}
\title{Saves Summaries Every N Steps}
\usage{
hook_summary_saver(
  save_steps = NULL,
  save_secs = NULL,
  output_dir = NULL,
  summary_writer = NULL,
  scaffold = NULL,
  summary_op = NULL
)
}
\arguments{
\item{save_steps}{An integer indicating saving summaries every N steps. Exactly one of
\code{save_secs} and \code{save_steps} should be set.}

\item{save_secs}{An integer indicating saving summaries every N seconds.}

\item{output_dir}{The directory to save the summaries to. Only used
if no \code{summary_writer} is supplied.}

\item{summary_writer}{The summary writer. If \code{NULL} and an \code{output_dir} was
passed, one will be created accordingly.}

\item{scaffold}{A scaffold to get summary_op if it's not provided.}

\item{summary_op}{A tensor of type \code{tf$string} containing the serialized
summary protocol buffer or a list of tensors. They are most likely an
output by TensorFlow summary methods like \code{tf$summary$scalar} or
\code{tf$summary$merge_all}. It can be passed in as one tensor; if more than
one, they must be passed in as a list.}
}
\description{
Saves Summaries Every N Steps
}
\seealso{
Other session_run_hook wrappers: 
\code{\link{hook_checkpoint_saver}()},
\code{\link{hook_global_step_waiter}()},
\code{\link{hook_history_saver}()},
\code{\link{hook_logging_tensor}()},
\code{\link{hook_nan_tensor}()},
\code{\link{hook_progress_bar}()},
\code{\link{hook_step_counter}()},
\code{\link{hook_stop_at_step}()},
\code{\link{session_run_hook}()}
}
\concept{session_run_hook wrappers}
