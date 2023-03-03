library(tidyverse)
library(randomForest)
library(caret)
library(OptimalCutpoints)
library(parallel)
library(doParallel)
library(ggplot2)

train_model <- function(data, predictors, outcome_var, method = "rf", num_clusters = detectCores(), 
                        num_folds = 5, opt_metric = "ROC", seed = 69) {
  
  data <- data %>% 
    select(all_of(c(predictors, outcome_var)))
  
  # Set up clusters - this speeds up cv training (4 clusters ~ twice as fast)
  cl <- makePSOCKcluster(num_clusters)
  registerDoParallel(cl)
  
  # Set up caret CV options
  ctrl <- trainControl(method = "repeatedcv",
                       number = num_folds,
                       classProbs = TRUE,
                       summaryFunction = twoClassSummary,
                       sampling = NULL,
                       verboseIter = TRUE)
  set.seed(seed)
  
  # The new big loop
  cv <- train(x = data %>% select(-{{outcome_var}}) %>% as.data.frame(),
              y = data[[outcome_var]],
              method = method,
              opt_metric = opt_metric,
              trControl = ctrl,
              tuneGrid = expand.grid(mtry = 1:((ncol(data) - 1))))
  
  # When you are done:
  stopCluster(cl)
  
  # print results
  print(cv)
  
  # get final model
  final_model <- cv$finalModel
  
  return(final_model)
  
}

train_multiclass_model <- function(data, predictors, outcome_var, method = "rf", num_clusters = detectCores(), 
                                   num_folds = 5, opt_metric = "ROC", seed = 69) {
  
  data <- data %>% 
    select(all_of(c(predictors, outcome_var)))
  
  # Set up clusters - this speeds up cv training (4 clusters ~ twice as fast)
  cl <- makePSOCKcluster(num_clusters)
  registerDoParallel(cl)
  
  # Set up caret CV options
  ctrl <- trainControl(method = "repeatedcv",
                       number = num_folds,
                       classProbs = TRUE,
                       summaryFunction = multiClassSummary,
                       sampling = NULL, # account for class imbalance
                       verboseIter = TRUE)
  
  set.seed(seed)
  
  # Calculate class weights for use in the model
  class_weights <- table(data[[outcome_var]]) %>% prop.table() %>% as.data.frame() %>% 
    setNames(c("class", "weight")) %>%
    mutate(weight = (1/weight)/3) %>% 
    right_join(train_df, by = c("class" = outcome_var)) %>% .[["weight"]]
  
  # The new big loop
  cv <- train(x = data %>% select(-{{outcome_var}}) %>% as.data.frame(),
              y = data[[outcome_var]],
              method = method,
              opt_metric = opt_metric,
              trControl = ctrl,
              tuneGrid = expand.grid(mtry = 1:((ncol(data) - 1))),
              weights = class_weights)
  
  # When you are done:
  stopCluster(cl)
  
  # print results
  print(cv)
  
  return(cv)
  
}

rf_cutoff_select <- function(rf_model, cut_method = "MaxKappa", CFN, CFP){
  
  # Use out-of-bag probabilities from final model in CV training to pick optimal cutoff
  prob <- rf_model$votes %>% as.data.frame %>% mutate(response = rf_model$y)
  
  cut <- optimal.cutpoints(X = names(prob)[1], status = "response", 
                           tag.healthy = names(prob)[2], methods = cut_method,
                           CFN = CFN, CFP = CFP, data = prob)
  
  cut <- cut$MaxKappa$Global$optimal.cutoff$cutoff
  
  cutoff <- c(cut, 1 - cut) # This order because we've called X = names(prob)[1] above, i.e. order from the rf_model object
  
  cat("Cutoff point chosen at (", names(prob)[1], ", ", names(prob)[2], ") = ", cutoff[1], ", ", cutoff[2], sep = "")
  
  rf_model$forest$cutoff <- cutoff
  
  return(list(rf_model = rf_model,
              cutoff = cutoff))
  
}

perform_rfe <- function(data, k, opt_metric, maximise, steps = NULL, outcome_var, predictors) {
  
  data <- data %>% select(all_of(c(predictors, outcome_var)))
  
  # Set up clusters - this speeds up cv training (4 clusters ~ twice as fast)
  cl <- makePSOCKcluster(detectCores())
  registerDoParallel(cl)
  
  # Set up caret CV options
  rfFuncs$summary <- twoClassSummary
  
  control <- rfeControl(functions = rfFuncs,
                        method = "cv",
                        number = k,
                        verbose = TRUE,
                        rerank = FALSE)
  
  trainctrl <- trainControl(classProbs= TRUE,
                            summaryFunction = twoClassSummary)
  
  if (is.null(steps))
    steps <- seq(1, (ncol(data) - 1), 1)
  
  x <- data %>% select(-{{outcome_var}}) %>% as.data.frame()
  y <- data[[outcome_var]]

  rfe.train <- rfe(x, y,
                   steps,
                   opt_metric = ifelse(is.factor(y), opt_metric, "RMSE"), # was "Kappa" for factors
                   maximise = ifelse(is.factor(y), maximise, FALSE),
                   rfeControl = control,
                   trControl = trainctrl)

  # When you are done:
  stopCluster(cl)

  # Spit out the results
  print(rfe.train)
  print(plot(rfe.train, type = c("g", "o"), cex = 1.0, col = 1:(ncol(data) - 1)))
  opt_predictors <- predictors(rfe.train) # call the predictors something

  # Try and make some silly graphs
  rfe.imp <- subset(as.data.frame(rfe.train$variables), Variables == (ncol(data) - 1)) # was 20
  rfe.imp$var <- as.factor(rfe.imp$var)
  rfe.imp$var <- reorder(rfe.imp$var, rfe.imp$Overall, mean)

  rfe.plot <- ggplot(rfe.imp, aes(x = var, y = Overall)) +
    ggtitle("Variable importance measures from cross validation runs") +
    xlab("") +
    ylab("Importance") +
    geom_violin() +
    theme_bw() +
    stat_summary(fun.y = mean, geom = "point", size = 1, color = "blue") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1), plot.title = element_text(hjust = 0.5))

  print(rfe.plot)

  # Define the optimal set of data from variable selection along with the response variable at the end
  data <- data %>% select(all_of(c(opt_predictors, outcome_var)))
  return(data)
  
}

