# Read data file

X <- read.table('gbm_input.csv', header = TRUE, sep = ',')

# Gradient boost it

library(gbm)

for (index in 1:20)
{
  # Work out which bits need to be predicted
  test <- is.na(X[,index])
  train <- !test
  # Go GBM!
  my_gbm <- gbm.fit(X[train,21:dim(X)[2]], X[train,index], n.trees=10000, distribution="gaussian", interaction.depth=3, shrinkage=0.01)
  X[test,index] <- predict.gbm(my_gbm,X[test,21:dim(X)[2]],10000)
}

# Write output

write.table(X, 'gbm_output.csv', sep = ',', row.names = FALSE)


