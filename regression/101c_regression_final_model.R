# library imports
library(tidymodels)
library(tidyverse)

# read in training and test data to predict log_total
train <- read_csv("train.csv")
test <- read_csv("test.csv")

# drop redundant column order_totals
train <- train %>% select(!order_totals) 

# random forest regression model
# hyperparameters: mtry = 2, trees = 1000, regularization.factor = 0.898
rf_spec <- rand_forest(mtry = 2, trees = 1000) %>% 
  set_engine("ranger", regularization.factor = .898) %>% 
  set_mode("regression")

# simple recipe to predict log_total from all other predictor columns
simple_rec <- recipe(log_total~., data = train)

# combine model and recipe to create workflow
wflow <- workflow() %>% 
  add_model(rf_spec) %>% 
  add_recipe(simple_rec)

# fit model to our training data
amz_rf_fit <- wflow %>% fit(data = train)

# use fitted model to predict on unknown test data
amz_rf_test_res <- predict(amz_rf_fit, new_data = test)

# add ID column back in to label predictions
amz_rf_test_res <- amz_rf_test_res %>% 
  bind_cols(id = test$id) %>% 
  relocate(id) %>% 
  rename(log_total = .pred)

# create csv file of predictions to submit for final model
write_csv(amz_rf_test_res, "rf_preds_reg_trees.csv")
