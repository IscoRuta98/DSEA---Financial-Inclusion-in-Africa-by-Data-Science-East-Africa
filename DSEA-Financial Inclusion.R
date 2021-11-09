rm(list = ls())
setwd("~/ZINDI/Hackathons/DSEA")

library(readr)
library(corrplot)
library(caret)
library(dplyr)
library(catboost)
library(gbm)
library(tree)
library(randomForest)
library(ranger)
library(pROC)
library(cluster)
library(NbClust)
library(fpc)
library(factoextra)
library(kohonen)
library(DMwR)
library(ROSE)


# Read in the data
Train_data = read_csv("Train_v2.csv")
Test = read_csv("Test_v2.csv")
Test$bank_account = numeric(length = nrow(Test))

whole_data = rbind(Train_data,Test)
uniqueid_whole_data = whole_data$uniqueid
whole_data = whole_data[,-3]

#Change character variables into categorical variables
whole_data$country = as.factor(whole_data$country)
whole_data$location_type = as.factor(whole_data$location_type)
whole_data$cellphone_access = as.factor(whole_data$cellphone_access)
whole_data$gender_of_respondent = as.factor(whole_data$gender_of_respondent)
whole_data$relationship_with_head = as.factor(whole_data$relationship_with_head)
whole_data$marital_status = as.factor(whole_data$marital_status)
whole_data$education_level = as.factor(whole_data$education_level)
whole_data$job_type = as.factor(whole_data$job_type)
whole_data$year = as.factor(whole_data$year)

whole_data$bank_account = as.factor(whole_data$bank_account)
whole_data$bank_account = as.numeric(whole_data$bank_account) -2
# 0= N0, 1 = Yes

#### Feature Engineering ####
whole_data$age_of_respondent = as.numeric(scale(whole_data$age_of_respondent))
whole_data$household_size = as.numeric(scale(whole_data$household_size))

# Create a new feature
whole_data$average_age = as.numeric(whole_data$age_of_respondent / whole_data$household_size)

#Change factor variable to numerical variable.
whole_data$location_type = as.numeric(whole_data$location_type)
whole_data$cellphone_access = as.numeric(whole_data$cellphone_access)
whole_data$gender_of_respondent = as.numeric(whole_data$gender_of_respondent)
whole_data$relationship_with_head = as.numeric(whole_data$relationship_with_head)
whole_data$marital_status = as.numeric(whole_data$marital_status)
whole_data$education_level = as.numeric(whole_data$education_level)
whole_data$job_type = as.numeric(whole_data$job_type)

Train_data = whole_data[1:nrow(Train_data),]
Test       = whole_data[nrow(Train_data)+1:nrow(Test),]


#### EDA ####
prop.table(table(Train_data$bank_account)) 
#Highly imbalanced data. 86% don't have a bank account, 14% do.

#### SMOTE ALGORITHM ###

## Smote : Synthetic Minority Oversampling Technique To Handle Class Imbalancy In Binary Classification

balance_data = ROSE(bank_account ~ ., data = Train_data, seed = 1998)$data
prop.table(table(balance_data$bank_account)) 


#### Splitting the data ####
set.seed(1998)
idx = sample(x = 1:nrow(balance_data),size = 0.8*nrow(balance_data),replace = F)

train = balance_data[idx,]
valid = balance_data[-idx,]

#### CAT BOOST ####
y_train = unlist(train[c('bank_account')])
X_train = train %>% select(-bank_account)
y_valid = unlist(valid[c('bank_account')])
X_valid = valid %>% select(-bank_account)

train_pool = catboost.load_pool(data = X_train, label = y_train)
valid_pool = catboost.load_pool(data = X_valid, label = y_valid)

params = list(iterations=500,
               learning_rate=0.01,
               depth=10,
               loss_function='Logloss',
               eval_metric='Logloss',
               random_seed = 55,
               od_type='Iter',
               metric_period = 50,
               od_wait=20,
               use_best_model=TRUE)

model.2 = catboost.train(learn_pool = train_pool,params = params)

pred.10 = catboost.predict(model.2,valid_pool,prediction_type = "Class")
mean(pred.10 != valid$bank_account)
1 - sum(diag(table(pred.10, valid$bank_account)))/length(valid$bank_account)


y_test = unlist(Test[c('bank_account')])
X_test = Test %>% select(-bank_account)

test_pool = catboost.load_pool(data = X_test, label = y_test)

catboost.2 = catboost.predict(model.2,test_pool,prediction_type = "Class")
sub9 = data.frame(paste(uniqueid_whole_data[nrow(Train_data)+1:nrow(Test)],Test$country,sep = " x "),catboost.2)
colnames(sub9) = c("uniqueid","bank_account")
write_csv(sub9, "~/ZINDI/Hackathons/DSEA/sub9.csv")
