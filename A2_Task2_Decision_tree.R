library(dplyr)
library(tibble)
library(corrplot)
library(rpart)
library(ROCR)

#Load data
data_raw <- read.csv("./A1_BC_SEER_data.csv")
data_temp <- data_raw

#Create new column
data_temp <- data_temp %>% mutate(survived = case_when(Survival.months >= 60 ~ 1, TRUE ~ 0 ))

#Remove irrelevant attributes
data <- data_temp %>% select(-Survival.months,-SEER.registry,-No., -Year.of.birth, -Year.of.diagnosis, -Patient.ID, -PR.Status.Recode.Breast.Cancer..1990..)

#Correlation plot
corrplot(cor(data), type = "upper")

#Convert as factor for categorical variables
for(name in names(data)) # For every column,
  {
    if (name != "Age.at.diagnosis")
    data[[name]] <- as.factor(data[[name]])
  }


################################################################################
#############################  Cross Validation #############################


#Cross Validation k=10
obs <- nrow(data)
k <- 10
tail <- obs%/%k
set.seed(10)
r <- rank(runif(obs))
block <- as.factor((r-1)%/%tail + 1)

#initialise vars
all.acc <-numeric(0)
all.auc <-numeric(0)

for (fold in 1:k)
{
  fit_k <-rpart(survived ~ ., data = data[block !=fold,], method = "class" )
  
  predict_k <- predict(fit_k, newdata = data[block==fold,],type = "class")
  
  conf_mat_k <- table(predict_k,
                      data$survived[block==fold])
  
  accuracy_k <- sum(diag(conf_mat_k))/sum(conf_mat_k)
  
  all.acc <- rbind(all.acc,accuracy_k)
  
  dt.kval <- prediction(as.numeric(predict_k), data[block==fold,]$survived)
  auc_k <-performance(dt.kval, measure = "auc")@y.values[[1]]
  
  
  all.auc <- rbind(all.auc, auc_k)
  
}

k_fold.auc <- mean(all.auc)
k_fold.acc <- mean(all.acc)

######################
#Cross Validation k=5
obs <- nrow(data)
k <- 5
tail <- obs%/%k
set.seed(10)
r <- rank(runif(obs))
block <- as.factor((r-1)%/%tail + 1)

all.acc.k5 <-numeric(0)
all.auc.k5 <-numeric(0)

for (fold in 1:k)
{
  fit_k5 <-rpart(survived ~ ., data = data[block !=fold,], method = "class" )
  
  predict_k5 <- predict(fit_k5, newdata = data[block==fold,],type = "class")
  
  conf_mat_k5 <- table(predict_k5,
                       data$survived[block==fold])
  
  accuracy_k5 <- sum(diag(conf_mat_k5))/sum(conf_mat_k5)
  
  all.acc.k5 <- rbind(all.acc,accuracy_k5)
  
  dt.kval.k5 <- prediction(as.numeric(predict_k5), data[block==fold,]$survived)
  auc_k5 <-performance(dt.kval.k5, measure = "auc")@y.values[[1]]
  
  
  
  all.auc.k5 <- rbind(all.auc.k5, auc_k5)
  
}

k_fold.auc.k5 <- mean(all.auc.k5)
k_fold.acc.k5 <- mean(all.acc.k5)

######################
#Cross Validation k=12
obs <- nrow(data)
k <- 12
tail <- obs%/%k
set.seed(10)
r <- rank(runif(obs))
block <- as.factor((r-1)%/%tail + 1)

all.acc.k12 <-numeric(0)
all.auc.k12 <-numeric(0)

for (fold in 1:k)
{
  fit_k12 <-rpart(survived ~ ., data = data[block !=fold,], method = "class" )
  
  predict_k12 <- predict(fit_k12, newdata = data[block==fold,],type = "class")
  
  conf_mat_k12 <- table(predict_k12,
                        data$survived[block==fold])
  
  accuracy_k12 <- sum(diag(conf_mat_k12))/sum(conf_mat_k12)
  
  all.acc.k12 <- rbind(all.acc,accuracy_k12)
  
  dt.kval.k12 <- prediction(as.numeric(predict_k12), data[block==fold,]$survived)
  auc_k12 <-performance(dt.kval.k12, measure = "auc")@y.values[[1]]
  
  
  
  all.auc.k12 <- rbind(all.auc.k12, auc_k12)
  
}

k_fold.auc.k12 <- mean(all.auc.k12)
k_fold.acc.k12 <- mean(all.acc.k12)

###################################################################################
########################  Hold Out ################################################

#Fix Seed
set.seed(7)

#Split data into training and test set
set_sample <- sample(1:nrow(data),nrow(data)/2,replace=FALSE)
data.train = data[set_sample,]
data.test = data[-set_sample,]

#model
fit <-rpart(survived ~ ., data = data.train, method = "class")

pred_holdout<- predict(fit, data.test,type = "class",decision.values = TRUE)

conf_mat <- table(pred_holdout,
                  data.test[,"survived"])

accuracy <- sum(diag(conf_mat))/sum(conf_mat)


#AUC
pred_holdout_roc<- predict(fit, data.test,type = "prob",decision.values = TRUE)[,2]
dt.holdout <- prediction(pred_holdout_roc, data.test$survived)
dt.perf.auc <-performance(dt.holdout, measure = "auc")@y.values[[1]]

####################################################################################
############################ Train, Test, Validation Split ########################

###### 60, 20, 20 Split #######
#Split data into training, test set, and validation set 60,20,20
index <- sample(seq(1, 3), size = nrow(data), replace = TRUE, prob = c(.6, .2, .2))
data.train.2 <- data[index == 1,]
data.test.2 <- data[index == 2,]
data.val.2 <- data[index == 3,]


#Training
fit_train_2 <-rpart(survived ~ ., data = data.train.2, method = "class")

conf_mat_train_2 <- table(predict(fit_train_2, data.train.2,type = "class"),
                  data.train.2[,"survived"])

accuracy_train_2 <- sum(diag(conf_mat_train_2))/sum(conf_mat_train_2)

#Validation

conf_mat_val_2 <- table(predict(fit_train_2, data.val.2,type = "class"),
                          data.val.2[,"survived"])

accuracy_val_2 <- sum(diag(conf_mat_val_2))/sum(conf_mat_val_2)


#Test

pred_ttv <- predict(fit_train_2, data.test.2,type = "class")

conf_mat_test_2 <- table(pred_ttv,
                         data.test.2[,"survived"])

accuracy_test_2 <- sum(diag(conf_mat_test_2))/sum(conf_mat_test_2)


#AUC
pred_ttv_roc <- predict(fit_train_2, data.test.2,type = "prob")[,2]
dt.ttv.622 <- prediction(as.numeric(pred_ttv_roc), data.test.2$survived)
dt.ttv.perf.auc <-performance(dt.ttv.622, measure = "auc")@y.values[[1]]

###### 70, 20, 10 Split #######
#Split data into training, test set, and validation set 60,20,20
index <- sample(seq(1, 3), size = nrow(data), replace = TRUE, prob = c(.7, .2, .1))
data.train.3 <- data[index == 1,]
data.test.3 <- data[index == 2,]
data.val.3 <- data[index == 3,]


#Training
fit_train_3 <-rpart(survived ~ ., data = data.train.3, method = "class")

conf_mat_train_3 <- table(predict(fit_train_3, data.train.3,type = "class"),
                          data.train.3[,"survived"])

accuracy_train_3 <- sum(diag(conf_mat_train_3))/sum(conf_mat_train_3)

#Validation

conf_mat_val_3 <- table(predict(fit_train_3, data.val.3,type = "class"),
                        data.val.3[,"survived"])

accuracy_val_3 <- sum(diag(conf_mat_val_3))/sum(conf_mat_val_3)


#Test

pred_ttv_731 <- predict(fit_train_3, data.test.3,type = "class")

conf_mat_test_3 <- table(pred_ttv_731,
                         data.test.3[,"survived"])

accuracy_test_3 <- sum(diag(conf_mat_test_3))/sum(conf_mat_test_3)


#AUC
pred_ttv_731_roc <- predict(fit_train_3, data.test.3,type = "prob")[,2]
dt.ttv.731 <- prediction(as.numeric(pred_ttv_731_roc), data.test.3$survived)
dt.ttv.731.perf.auc <-performance(dt.ttv.731, measure = "auc")@y.values[[1]]

###############  ROC Curve ######################


dt.holdout.roc <- performance(dt.holdout, measure = "tpr", x.measure = "fpr")
dt.ttv.622.roc <- performance(dt.ttv.622, measure = "tpr", x.measure = "fpr")
dt.ttv.721.roc <- performance(dt.ttv.731, measure = "tpr", x.measure = "fpr")


dev.new()
plot(dt.holdout.roc, col = 3)
plot(dt.ttv.622.roc, add = TRUE, col = 2)
plot(dt.ttv.721.roc, add = TRUE, col = 4)
abline(0, 1, lty=3)
legend("bottomright", c("Hold Out","Train-Test-Validation - 60-20-20","Train-Test-Validation - 70-20-10"), lty = 1,
       col = 4:2)

#################################################
  
  
  
