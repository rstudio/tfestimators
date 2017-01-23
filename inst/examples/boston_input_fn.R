library(tensorflow)

tf$logging$set_verbosity(tf$logging$INFO)

boston_train <- read.csv("inst/extdata/boston_train.csv", sep = ",")
boston_test <- read.csv("inst/extdata/boston_test.csv", sep = ",")
boston_predict <- read.csv("inst/extdata/boston_predict.csv", sep = ",")

temp_model_dir <- tempfile()
dir.create(temp_model_dir)

COLUMNS <- c("crim", "zn", "indus", "nox", "rm", "age",
             "dis", "tax", "ptratio", "medv")
FEATURES <- COLUMNS[-length(COLUMNS)]
LABEL <- COLUMNS[length(COLUMNS)]

input_fn <- function(data_set) {
  
  feature_cols <- lapply(1:length(FEATURES), function(i) {
    tf$constant(data_set[, i])
  })
  names(feature_cols) <- FEATURES
  
  if(ncol(data_set) >= length(COLUMNS)) {
    labels <- tf$constant(data_set[, length(COLUMNS)])
  } else {
    # boston_test does not contain labels
    labels <- NA 
  }
  return(list(feature_cols, labels))
}

feature_cols <- lapply(FEATURES, function(feature) {tf$contrib$layers$real_valued_column(feature)})

# Build 2 layer fully connected DNN with 10, 10 units respectively.
regressor <- tf$contrib$learn$DNNRegressor(feature_columns = feature_cols,
                                           hidden_units = c(10L, 10L),
                                           model_dir = temp_model_dir)

# Fit
regressor$fit(input_fn = function(){input_fn(boston_train)}, steps = 100)

# Score accuracy
ev <- regressor$evaluate(input_fn = function(){input_fn(boston_test)}, steps=1)
loss_score <- ev$loss
print(paste0("Loss score is: ", loss_score))

# Generate predictions
predictions <- regressor$predict(input_fn = function(){input_fn(boston_predict)})
predictions <- iterate(predictions)
