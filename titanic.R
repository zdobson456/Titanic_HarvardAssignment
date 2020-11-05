install.packages('titanic')
library(titanic)    # loads titanic_train data frame
library(caret)
library(tidyverse)
library(rpart)

# 3 significant digits
options(digits = 3)

# clean the data - `titanic_train` is loaded with the titanic package
titanic_clean <- titanic_train %>%
  mutate(Survived = factor(Survived),
         Embarked = factor(Embarked),
         Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age), # NA age to median age
         FamilySize = SibSp + Parch + 1) %>%    # count family members
  select(Survived,  Sex, Pclass, Age, Fare, SibSp, Parch, FamilySize, Embarked)


# Splits titanic_clean into test and training sets with a 20 % data partition on the survived column. Assigns the 20% to test_set and 80% to train set
set.seed(42)
test_index <- createDataPartition(titanic_clean$Survived, times = 1, p=0.2, list = FALSE)

train_set <- titanic_clean[-test_index,]
test_set <- titanic_clean[test_index,]

nrow(train_set)
nrow(test_set)

mean(train_set$Survived == 1)

#Baseline prediction by guessing the outcome
set.seed(3)
guess <- sample(c(0,1), nrow(test_set), replace = TRUE)
test_set %>%
  filter(Survived == guess) %>%
  summarize(n() / nrow(test_set))
# accuracy = .542

# Predicting survival by sex on training set
train_set %>%
  group_by(Sex) %>%
  summarize(Survived = mean(Survived == 1))

sex_model <- ifelse(test_set$Sex == "female", 1, 0)    # predict Survived=1 if female, 0 if male
mean(sex_model == test_set$Survived)    # calculate accuracy
# Female .81
# Male .19

# Predicting survival by sex: if the survival rte for a sex is over .5 predict survival for all individuals of that sex. Death if under
test_set %>%
  summarize((sum(Sex == 'female' & Survived == 1) + sum(Sex == 'male' & Survived == 0))/ n())

# Accuracy: .81

###########################

# Predicting survival by passenger class in training set
train_set %>%
  group_by(Pclass) %>%
  summarize(Survived = mean(Survived == 1))

# First class was more likely to survive than die

# Predicting survival using passenger class on the test set. if survival rate for a class is over 0.5 predict survival, death if otherwise.
class_model <- ifelse(test_set$Pclass == 1, 1, 0)    # predict survival only if first class
mean(class_model == test_set$Survived)    # calculate accuracy

survival_class <- titanic_clean %>%
  group_by(Pclass) %>%
  summarize(PredictingSurvival = ifelse(mean(Survived == 1) >=0.5, 1, 0))
survival_class

survival_class <- titanic_clean %>%
  group_by(Sex, Pclass) %>%
  summarize(PredictingSurvival = ifelse(mean(Survived == 1) > 0.5, 1, 0))
survival_class

# Class and sex most likely to survive: female in 1st & 2nd class

# Predicting survival using both sex and passenger on the test set if both sex/class combination is over .5 predict survival, otherwise death
test_set %>%
  inner_join(survival_class, by ='Pclass') %>%
  summarize(PredictingSurvival = mean(Survived == PredictingSurvival))

class_model <- ifelse(test_set$Pclass == 1, 1, 0)    # predict survival only if first class
mean(class_model == test_set$Survived) 

train_set %>%
  group_by(Sex & inner_join(survival_class, by = 'Pclass')) %>%
  summarize(PredictingSurvival = mean(Survived == PredictingSurvival))

train_set %>%
  group_by(Sex, Pclass) %>%
  summarize(Survived = mean(Survived == 1)) %>%
  filter(Survived > 0.5)

test_set %>%
  inner_join(survival_class, by=c('Sex', 'Pclass')) %>%
  summarize(PredictingSurvival = mean(Survived == PredictingSurvival))

sex_class_model <- ifelse(test_set$Sex == "female" & test_set$Pclass != 3, 1, 0)
mean(sex_class_model == test_set$Survived)

# Accuracy of sex and class based prediction methods on test .793

# confusing Matrix analysis

sex_model <- train_set %>%
  group_by(Sex) %>%
  summarize(Survived_predict = ifelse(mean(Survived == 1) > 0.5, 1, 0))

test_set1 <- test_set %>%
  inner_join(sex_model, by = 'Sex')

cm1 <- confusionMatrix(data = factor(test_set1$Survived_predict), reference = factor(test_set1$Survived))

class_model <- train_set %>%
  group_by(Pclass) %>%
  summarize(Survived_predict = ifelse(mean(Survived == 1) != 3, 1, 0))

test_set2 <- train_set %>%
  inner_join(class_model, by = 'Pclass')

cm2 <- confusionMatrix(data = factor(test_set2$Survived_predict), reference = factor(test_set2$Survived))
cm2

sex_class_model <- train_set %>%
  group_by(Sex, Pclass) %>%
  summarize(Survived_predict = ifelse(mean(Survived == 1) > 0.5, 1, 0))

test_set3 <- test_set %>%
  inner_join(sex_class_model, by=c('Sex', 'Pclass'))

cm3 <- confusionMatrix(data = factor(test_set3$Survived_predict), reference = factor(test_set3$Survived))

cm1
cm2
cm3

F_meas(data=factor(test_set1$Survived), reference = factor(test_set1$Survived_predict))
F_meas(data=factor(test_set2$Survived), reference = factor(test_set2$Survived_predict))
F_meas(data=factor(test_set3$Survived), reference = factor(test_set3$Survived_predict))

# The sex and class model had the highest sensitivity
# Sex only had the highest specificity
# Sex only had the highest balanced accuracy

# The maximum value of balanced accuracy is .806

# Maximum value of the F1 score is .872


# Training LDA and QDA with caret using fare as the only predictor
set.seed(1)

fit_lda <- train(Survived ~ Fare, data = train_set, method = 'lda')
Survived_hat <- predict(fit_lda, test_set)
mean(test_set$Survived == Survived_hat)

fit_qda <- train(Survived ~ Fare, data = train_set, method = 'qda')
Survived_hat <- predict(fit_qda, test_set)
mean(test_set$Survived == Survived_hat)

# Accuracy on the test set for the LDA Model: .659
# Accuracy on the test set for the QDA Model: .665


# Logistic regression model with caret glm method using age as the only predictor
set.seed(1)
fit_logreg_a <- glm(Survived ~ Age, data = train_set, family = 'binomial')
survived_hat_a <- ifelse(predict(fit_logreg_a, test_set) >= 0, 1, 0)
mean(survived_hat_a == test_set$Survived)

# Accuracy .615

# Logistic regression model with caret glm method using four predictors: sex, class, fare, and age.
set.seed(1)
fit_logreg_b <- glm(Survived ~ Sex + Pclass + Fare + Age, data = train_set, family = 'binomial')
survived_hat_b <- ifelse(predict(fit_logreg_b, test_set) >= 0, 1, 0)
mean(survived_hat_b == test_set$Survived)

# Accuracy .821

# Logistic regression model with caret gml method using all predictors
set.seed(1)
str(train_set)
fit_logreg_c <- glm(Survived ~ ., data = train_set, family = 'binomial')
survived_hat_c <- ifelse(predict(fit_logreg_c, test_set) >= 0, 1, 0)
mean(survived_hat_c == test_set$Survived)

# Accuracy .821

# KNN model on the training set using caret train function
set.seed(6, sample.kind = "Rounding") # if using R 3.6 or later
train_knn <- train(Survived ~ .,
                   method = "knn",
                   data = train_set,
                   tuneGrid = data.frame(k = seq(3, 51, 2)))
train_knn$bestTune

# Optimal value for the number of neighbors 25

# Ploting the KNN model 
ggplot(train_knn)

# Accuracy of the KNN model on the test set
knn_preds <- predict(train_knn, test_set)
mean(knn_preds == test_set$Survived)

# Accuracy .732

# 10-fold cross validation 
set.seed(8, sample.kind = "Rounding")
fit_knn10 <- train(Survived ~ ., 
                   data=train_set, 
                   method = "knn",
                   tuneGrid = data.frame(k = seq(3, 51, 2)),
                   trControl = trainControl(method = "cv", number=10, p=0.9))
fit_knn10
survived_hat <- predict(fit_knn10, test_set)
cm_test <- confusionMatrix(data = survived_hat, reference = test_set$Survived)
cm_test$overall["Accuracy"]

# Optimal value of K is 5
# accuracy is .692

# Classification tree model using caret rpart method
set.seed(10, sample.kind = "Rounding")    # simulate R 3.5
train_rpart <- train(Survived ~ ., 
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.05, 0.002)),
                     data = train_set)
train_rpart$bestTune

rpart_preds <- predict(train_rpart, test_set)
mean(rpart_preds == test_set$Survived)

# Optimal value of the complexity parameter: .016
# accuracy of the decision tree model on the test set: .849

# graph decision tree
fit_rpart11$finalModel
plot(fit_rpart11$finalModel, margin=0.1)
text(fit_rpart11$finalModel, cex = 0.75)

# Random forest model
set.seed(14)
train_rf <- train(Survived ~.,
                  method = "rf",
                  tuneGrid = data.frame(mtry = seq(1:7)),
                  ntree = 100,
                  data = train_set)
train_rf$bestTune

rf_pred <- predict(train_rf, test_set)
mean(rf_pred == test_set$Survived)

varImp(train_rf)

# accuracy: .844




