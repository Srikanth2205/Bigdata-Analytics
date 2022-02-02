library(tidyverse) # metapackage with lots of helpful functions
library(caTools)
library(reshape2)
library(ROCR)
library(caret)
task1_data <- as.data.frame(read.csv('heart.csv'))
str(task1_data)
summary(task1_data)

#Splitting output variable into test and train sets
set.seed(99)
split = sample.split(task1_data$target, SplitRatio = 0.7)
train_data = subset(task1_data, split==TRUE)
test_data = subset(task1_data, split==FALSE)

#Logistic Regression Model
Model_logi = glm(target~ ï..age + sex  + cp + trestbps + chol + fbs + restecg + thalach + exang + oldpeak + slope + ca + thal, data=train_data, family='binomial')

summary(Model_logi)

# Checking multicollinearity 
check_linearity=abs(cor(train_data))>0.7

#View multicollinearity  in a heat map

melted_data = melt(abs(check_linearity))# melting data  to prepare heat map
head(melted_data)


ggplot(melted_data, aes(x = Var1, y=Var2, fill=as.numeric(value))) + geom_tile() +
  geom_text(aes(Var1, Var2, label=as.numeric(value)),color='black',size=2)+
  scale_color_gradient(low='blue',high='red') +
  theme(axis.text.x = element_text(angle=45, vjust=1, hjust=1))

# The figure demonstrates clearly that no variables are highly correlated (positive or negative).
#We will remove variables based on significance levels now that there is no multi-linearity among the variables.

summary(Model_logi)
Model_logi2 = glm(target~ ï..age + sex  + cp + trestbps + chol + restecg + thalach + exang + oldpeak + slope + ca + thal, 
                 data=train_data, family='binomial')
summary(Model_logi2)

Model_logi2 = glm(target~ ï..age + sex  + cp + trestbps + chol + restecg + exang + oldpeak + slope + ca + thal, 
                  data=train_data, family='binomial')
summary(Model_logi2)


Model_logi2 = glm(target~ ï..age + sex  + cp + trestbps + restecg + exang + oldpeak + slope + ca + thal, 
                  data=train_data, family='binomial')
summary(Model_logi2)
Model_logi2 = glm(target~  sex  + cp + trestbps + restecg + exang + oldpeak + slope + ca + thal, 
                  data=train_data, family='binomial')

summary(Model_logi2)

Model_logi2 = glm(target~  sex  + cp + trestbps + exang + oldpeak + slope + ca + thal, 
                  data=train_data, family='binomial')

summary(Model_logi2)


#Now all variables are significant in the model

#Predictions on Training Set
train_data_predict = predict(Model_logi2, type='response')


#Confusion matrix using threshold of 0.5
table(train_data$target, train_data_predict>0.5)

#Accuracy on training set
(80+104)/nrow(train_data)

#Predictions on Test set
test_data_predict = predict(Model_logi2, newdata=test_data, type='response')


#Confusion matrix using threshold of 0.5
table(test_data$target, test_data_predict>0.5)


#Accuracy
(27+46)/(nrow(test_data))

table(test_data$target, test_data_predict>0.2)

#Accuracy
(21+47)/(nrow(test_data))

#Plotting ROCR curve

Predict_ROCR = prediction(test_data_predict, test_data$target)


#Area under the curve
auc_curve = as.numeric(performance(Predict_ROCR, 'auc')@y.values)
auc_curve


Preformance_ROCR = performance(Predict_ROCR, 'tpr','fpr')
plot(Preformance_ROCR, colorize=TRUE, main='ROCR Curve')

trctrl <- trainControl(method = "cv", number = 10, savePredictions=TRUE)
nb_fit <- train(factor(target) ~., data = task1_data, method = "naive_bayes", trControl=trctrl, tuneLength = 0)
nb_fit

pred <- nb_fit$pred
pred$equal <- ifelse(pred$pred == pred$obs, 1,0)
eachfold <- pred %>%                                        
  group_by(Resample) %>%                         
  summarise_at(vars(equal),                     
               list(Accuracy = mean))              
eachfold

ggplot(data=eachfold, aes(x=Resample, y=Accuracy, group=1)) +
  geom_boxplot(color="maroon") +
  geom_point() +
  theme_minimal()


