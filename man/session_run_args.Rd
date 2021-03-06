% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/session_run_hooks_builtin_wrappers.R
\name{session_run_args}
\alias{session_run_args}
\title{Create Session Run Arguments}
\usage{
session_run_args(...)
}
\arguments{
\item{...}{A set of tensors or operations.}
}
\description{
Create a set of session run arguments. These are used as the return values in
the \code{before_run(context)} callback of a \code{\link[=session_run_hook]{session_run_hook()}}, for requesting
the values of specific tensor in the \code{after_run(context, values)} callback.
}
\seealso{
\code{\link[=session_run_hook]{session_run_hook()}}
}
