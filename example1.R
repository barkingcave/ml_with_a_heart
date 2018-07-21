
library(ggplot2)
library(dplyr)
library(e1071)
library(caret)
library(quanteda)
library(readr)
library(tidyverse)
library(keras)


select_values <- function(df) {
  # return matrix
  df %>% 
    select(slope_of_peak_exercise_st_segment,
           resting_blood_pressure,
           chest_pain_type,
           #num_major_vessels,
           fasting_blood_sugar_gt_120_mg_per_dl,
           #resting_ekg_results,
           #serum_cholesterol_mg_per_dl,
           sex,
           age,
           max_heart_rate_achieved,
           exercise_induced_angina) %>% 
    as.matrix()
}

train_values <- read_csv("train_values.csv")
train_labels <- read_csv("train_labels.csv")

test_values <- read_csv("test_values.csv")

train_v <- select_values(train_values)
test_v <- select_values(test_values)

dim(train_v)

# Create sequential model
model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(ncol(train_v))) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(model)

model %>% compile(
  optimizer = "rmsprop",
  #loss = "binary_crossentropy",
  loss = "categorical_crossentropy",
  metrics =c("accuracy"))

history <- model %>% fit(
  train_v, train_labels$heart_disease_present, 
  epochs = 16, batch_size = 256,
  validation_split = 0.2)

plot(history)

predictions <- model %>% predict_classes(test_v) 
p1 <- model %>% predict_proba(test_v) 
p_df <- p1 %>% 
  as_data_frame() %>% 
  cbind(patient_id = test_values$patient_id) %>% 
  rename(heart_disease_present = V1) %>% 
  select(patient_id, heart_disease_present) 

write_csv(p_df, "my_predictions.csv")

#model_1 <- model %>%
#  evaluate(x_test, y_test)
#model_1