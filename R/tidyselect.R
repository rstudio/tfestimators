tidyselect_data <- function() {
  tidyselect <- asNamespace("tidyselect")
  exports <- getNamespaceExports(tidyselect)
  data <- lapply(exports, function(export) {
    tidyselect[[export]]
  })
  names(data) <- exports
  data
}
