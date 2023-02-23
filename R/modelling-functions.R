library(tidyverse)
library(randomForest)
library(caret)
library(OptimalCutpoints)
library(parallel)
library(doParallel)

cross_validation <- function(x,
                             y,
                             algo = "rf", 
                             k = k, 
                             sampling = NULL,
                             metric = "ROC",
                             tGrid = NULL, 
                             seed = 69,
                             ...){
  
  # Set up clusters - this speeds up cv training (4 clusters ~ twice as fast)
  cl <- makePSOCKcluster(detectCores())
  registerDoParallel(cl)
  
  # Set up caret CV options
  ctrl <- trainControl(method = "repeatedcv", 
                       number = k,
                       classProbs = TRUE,
                       summaryFunction = twoClassSummary,
                       sampling = sampling, 
                       verboseIter = TRUE)
  set.seed(seed)
  
  
  # The new big loop
  cv1 <- train(x = x, y = y, 
               method = algo,
               metric = metric, 
               trControl = ctrl,
               tuneGrid = tGrid,
               ...)
  
  # When you are done:
  stopCluster(cl)
  
  return(cv1)
  
}

rf_cutoff_select <- function(rf_model, cut_method = "MaxKappa"){
  
  # Use out-of-bag probabilities from final model in CV training to pick optimal cutoff
  prob <- rf_model$votes %>% as.data.frame %>% mutate(response = rf_model$y)
  
  cut <- optimal.cutpoints(X = names(prob)[1], status = "response", 
                           tag.healthy = names(prob)[2], methods = cut_method, 
                           data = prob)
  
  cut <- cut$MaxKappa$Global$optimal.cutoff$cutoff
  
  cutoff <- c(cut, 1 - cut) # This order because we've called X = names(prob)[1] above, i.e. order from the rf_model object
  
  cat("Cutoff point chosen at (", names(prob)[1], ", ", names(prob)[2], ") = ", cutoff[1], ", ", cutoff[2], sep = "")
  
  rf_model$forest$cutoff <- cutoff
  
  return(rf_model)
  
}