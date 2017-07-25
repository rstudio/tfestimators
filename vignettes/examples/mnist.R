root <- rprojroot::find_package_root_file()

library(ggplot2)
library(reshape2)
library(tensorflow)
library(tfestimators)

# download the MNIST data sets, and read them into R
sources <- list(
  
  train = list(
    x = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    y = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
  ),
  
  test = list(
    x = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    y = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
  )
  
)

# read an MNIST file (encoded in IDX format)
read_idx <- function(file) {
  
  # create binary connection to file
  conn <- gzfile(file, open = "rb")
  on.exit(close(conn), add = TRUE)
  
  # read the magic number as sequence of 4 bytes
  magic <- readBin(conn, what = "raw", n = 4, endian = "big")
  ndims <- as.integer(magic[[4]])
  
  # read the dimensions (32-bit integers)
  dims <- readBin(conn, what = "integer", n = ndims, endian = "big")
  
  # read the rest in as a raw vector
  data <- readBin(conn, what = "raw", n = prod(dims), endian = "big")
  
  # convert to an integer vecto
  converted <- as.integer(data)
  
  # read as matrix (or drop dimension if needed)
  result <- t(matrix(converted, ncol = dims[[1]], nrow = prod(dims[-1])))
  
  if (ncol(result) == 1)
    return(result[, 1])
  
  result
}

dir <- file.path(root, "vignettes/examples/data")
dir.create(dir, recursive = TRUE, showWarnings = FALSE)
mnist <- rapply(sources, classes = "character", how = "list", function(url) {
  
  # download + extract the file at the URL
  target <- file.path(dir, basename(url))
  if (!file.exists(target))
    download.file(url, target)
  
  # read the IDX file
  read_idx(target)
  
})

# convert training data intensities to 0-1 range
mnist$train$x <- mnist$train$x / 255
mnist$test$x <- mnist$test$x / 255

# try plotting the pixel intensities for the first 16 numbers
data <- array(mnist$train$x[1:16, ], dim = c(16, 28, 28))
melted <- melt(data, varnames = c("image", "x", "y"), value.name = "intensity")
ggplot(melted, aes(x = x, y = y, fill = intensity)) +
  geom_tile() +
  scale_y_reverse() +
  facet_wrap(~ image) +
  scale_fill_continuous(low = "white", high = "black") +
  theme(
    strip.background = element_blank(),
    strip.text.x = element_blank(),
    panel.spacing = unit(0, "lines"),
    axis.text = element_blank(),
    axis.ticks = element_blank()
  ) +
  labs(
    title = "MNIST Image Data",
    subtitle = "Visualization of first 16 images contained in MNIST data set.",
    x = NULL,
    y = NULL
  )

# construct a linear classifier
classifier <- linear_classifier(
  feature_columns = feature_columns(
    column_numeric("x", shape = shape(784L))
  ),
  model_dir = tempfile("tensorflow-mnist-linear-classifier-"),
  n_classes = 10L  # 10 digits
)

# construct an input function generator
.input_fn <- function(data) {
  input_fn(data, response = "y", features = "x")
}

# train the classifier
train(classifier, input_fn = .input_fn(mnist$train), steps = 20)

# evaluate the classifier on the test dataset
evaluate(classifier, input_fn = .input_fn(mnist$test))
