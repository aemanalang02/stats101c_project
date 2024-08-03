library(dplyr)
library(tidymodels)
library(tidyverse)
library(ranger)

train <- read.csv("train_class.csv")
test <- read.csv("test_class.csv")

#Fixing missing economic values
va_missing_value_inputs <- train %>%
  filter(grepl(", Virginia", name), !is.na(income_per_cap_2016)) %>%
  summarise(income_2016 = mean(income_per_cap_2016), income_2017 = mean(income_per_cap_2017), income_2018 = mean(income_per_cap_2018), income_2019 = mean(income_per_cap_2019), income_2020 = mean(income_per_cap_2020), gdp16 = mean(gdp_2016), gdp17 = mean(gdp_2017), gdp18 = mean(gdp_2018), gdp19 = mean(gdp_2019), gdp20 = mean(gdp_2020))

train_missing_values <- train %>%
  filter(is.na(income_per_cap_2016)) %>%
  mutate(income_per_cap_2016 = va_missing_value_inputs$income_2016, income_per_cap_2017 = va_missing_value_inputs$income_2017, income_per_cap_2018 = va_missing_value_inputs$income_2018, income_per_cap_2019 = va_missing_value_inputs$income_2019, income_per_cap_2020 = va_missing_value_inputs$income_2020, gdp_2016 = va_missing_value_inputs$gdp16, gdp_2017 = va_missing_value_inputs$gdp17, gdp_2018 = va_missing_value_inputs$gdp18, gdp_2019 = va_missing_value_inputs$gdp19, gdp_2020 = va_missing_value_inputs$gdp20)

train <- train %>%
  filter(!is.na(income_per_cap_2016)) %>%
  bind_rows(train_missing_values) %>%
  arrange(id)

test_missing_values <- test %>%
  filter(is.na(income_per_cap_2016)) %>%
  mutate(income_per_cap_2016 = va_missing_value_inputs$income_2016, income_per_cap_2017 = va_missing_value_inputs$income_2017, income_per_cap_2018 = va_missing_value_inputs$income_2018, income_per_cap_2019 = va_missing_value_inputs$income_2019, income_per_cap_2020 = va_missing_value_inputs$income_2020, gdp_2016 = va_missing_value_inputs$gdp16, gdp_2017 = va_missing_value_inputs$gdp17, gdp_2018 = va_missing_value_inputs$gdp18, gdp_2019 = va_missing_value_inputs$gdp19, gdp_2020 = va_missing_value_inputs$gdp20)

test <- test %>%
  filter(!is.na(income_per_cap_2016)) %>%
  bind_rows(test_missing_values) %>%
  arrange(id)

#get rid of description variables
train <- train %>%
  dplyr::select(-id, -name, -total_votes) %>%
  mutate(winner = as.factor(winner))

#turning population estimates into proportions
train <- train %>%
  mutate(across(starts_with("C"), ~ .x / x0001e)) %>%
  mutate(across(starts_with("x"), ~ .x / x0001e)) %>%
  dplyr::select(-x0001e)

test <- test %>%
  mutate(across(starts_with("C"), ~ .x / x0001e)) %>%
  mutate(across(starts_with("x"), ~ .x / x0001e)) %>%
  dplyr::select(-x0001e)

#Feature Selection
rf_model_feat <- ranger(winner ~ ., data = train, importance = 'impurity')

importance <- importance(rf_model_feat)
var_importance <- data.frame(Feature = names(importance), Importance = importance)

threshold <- 0.01

selected_features <- var_importance %>%
  filter(Importance > threshold) %>%
  pull(Feature)

#Random Forest Model
rf_recipe_features <- recipe(winner ~ ., data = train) %>%
  update_role(all_of(selected_features), new_role = "predictor") %>%
  step_zv() %>%
  step_rm(which(duplicated(as.list(train)))) %>%
  step_normalize(all_predictors()) %>%
  step_dummy(all_nominal_predictors())

rf_model_features <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_workflow_features <- workflow() %>%
  add_recipe(rf_recipe_features) %>%
  add_model(rf_model_features)

rf_grid_features <- grid_regular(mtry(range = c(1, 10)), min_n(range = c(1, 10)), levels = 5)
set.seed(111)
rf_cv_folds_features <- vfold_cv(train, v = 5, strata = winner)

set.seed(111)
rf_tuned_results_features <- tune_grid(rf_workflow_features,
                              resamples = rf_cv_folds_features,
                              grid = rf_grid_features,
                              metrics = metric_set(accuracy))

rf_best_result_features <- select_best(rf_tuned_results_features)
final_rf_wkflw_features <- finalize_workflow(rf_workflow_features, rf_best_result_features)

set.seed(111)
rf_fit_features <- final_rf_wkflw_features %>%
  fit(data = train)

# Evaluate the Model
rf_predictions_features <- rf_fit_features %>%
  predict(new_data = test)

rf_predictions_features$id <- test$id
