tidyselect_data <- function() {
  tidyselect <- asNamespace("tidyselect")
  exports <- getNamespaceExports(tidyselect)
  mget(exports, tidyselect, inherits = TRUE)
}
