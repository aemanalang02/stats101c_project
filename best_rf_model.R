library(dplyr)
library(tidymodels)
library(tidyverse)

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

#Random Forest Model
rf_recipe <- recipe(winner ~ ., data = train) %>%
  step_rm(which(duplicated(as.list(train)))) %>%
  step_normalize(all_predictors()) %>%
  step_dummy(all_nominal_predictors())

rf_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_workflow <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(rf_model)

rf_grid <- grid_regular(mtry(range = c(1, 10)), min_n(range = c(1, 10)), levels = 5)
set.seed(999)
rf_cv_folds <- vfold_cv(train, v = 5, strata = winner)

set.seed(999)
rf_tuned_results <- tune_grid(rf_workflow,
                               resamples = rf_cv_folds,
                               grid = rf_grid,
                               metrics = metric_set(accuracy))

rf_best_result <- select_best(rf_tuned_results)
final_rf_wkflw <- finalize_workflow(rf_workflow, rf_best_result)

set.seed(999)
rf_fit <- final_rf_wkflw %>%
  fit(data = train)

# Evaluate the Model
rf_predictions <- rf_fit %>%
  predict(new_data = test)

rf_predictions$id <- test$id