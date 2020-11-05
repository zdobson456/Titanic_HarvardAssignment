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

set.seed(42)
test_index <- createDataPartition(titanic_clean$Survived, times = 1, p=0.2, list = FALSE)

train_set <- titanic_clean[-test_index,]
test_set <- titanic_clean[test_index,]

nrow(train_set)
nrow(test_set)

mean(train_set$Survived == 1)

set.seed(3)
guess <- sample(c(0,1), nrow(test_set), replace = TRUE)
test_set %>%
  filter(Survived == guess) %>%
  summarize(n() / nrow(test_set))

train_set %>%
  group_by(Sex) %>%
  summarize(Survived = mean(Survived == 1))

sex_model <- ifelse(test_set$Sex == "female", 1, 0)    # predict Survived=1 if female, 0 if male
mean(sex_model == test_set$Survived)    # calculate accuracy

test_set %>%
  summarize((sum(Sex == 'female' & Survived == 1) + sum(Sex == 'male' & Survived == 0))/ n())
###########################
train_set %>%
  group_by(Pclass) %>%
  summarize(Survived = mean(Survived == 1))

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

###########################################################################################################################################

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

################################################################################################################# PART 2 ####################
######### Question 6
set.seed(1)

fit_lda <- train(Survived ~ Fare, data = train_set, method = 'lda')
Survived_hat <- predict(fit_lda, test_set)
mean(test_set$Survived == Survived_hat)

fit_qda <- train(Survived ~ Fare, data = train_set, method = 'qda')
Survived_hat <- predict(fit_qda, test_set)
mean(test_set$Survived == Survived_hat)

######## harvard answer
#set.seed(1) # if using R 3.5 or earlier
set.seed(1, sample.kind = "Rounding") if using R 3.6 or later
train_qda <- train(Survived ~ Fare, method = "qda", data = train_set)
qda_preds <- predict(train_qda, test_set)
mean(qda_preds == test_set$Survived)

######### Question 7 
#a
set.seed(1)
fit_logreg_a <- glm(Survived ~ Age, data = train_set, family = 'binomial')
survived_hat_a <- ifelse(predict(fit_logreg_a, test_set) >= 0, 1, 0)
mean(survived_hat_a == test_set$Survived)

#b
set.seed(1)
fit_logreg_b <- glm(Survived ~ Sex + Pclass + Fare + Age, data = train_set, family = 'binomial')
survived_hat_b <- ifelse(predict(fit_logreg_b, test_set) >= 0, 1, 0)
mean(survived_hat_b == test_set$Survived)

#c
set.seed(1)
str(train_set)
fit_logreg_c <- glm(Survived ~ ., data = train_set, family = 'binomial')
survived_hat_c <- ifelse(predict(fit_logreg_c, test_set) >= 0, 1, 0)
mean(survived_hat_c == test_set$Survived)

#### Harvard Answers
#a
#set.seed(1) # if using R 3.5 or earlier
set.seed(1, sample.kind = "Rounding") if using R 3.6 or later
train_glm_age <- train(Survived ~ Age, method = "glm", data = train_set)
glm_preds_age <- predict(train_glm_age, test_set)
mean(glm_preds_age == test_set$Survived)

#b
#set.seed(1) # if using R 3.5 or earlier
set.seed(1, sample.kind = "Rounding") if using R 3.6 or later
train_glm <- train(Survived ~ Sex + Pclass + Fare + Age, method = "glm", data = train_set)
glm_preds <- predict(train_glm, test_set)
mean(glm_preds == test_set$Survived)

#c
#set.seed(1) # if using R 3.5 or earlier
set.seed(1, sample.kind = "Rounding") if using R 3.6 or later
train_glm_all <- train(Survived ~ ., method = "glm", data = train_set)
glm_all_preds <- predict(train_glm_all, test_set)
mean(glm_all_preds == test_set$Survived)

### Question 9
#a
set.seed(6)
k <- seq(3, 51, 2)
fit_knn <- train(Survived ~., data = train_set, method='knn', tuneGrid = data.frame(k))
fit_knn$bestTune

# Harvard Answer
set.seed(6, sample.kind = "Rounding") # if using R 3.6 or later
train_knn <- train(Survived ~ .,
                   method = "knn",
                   data = train_set,
                   tuneGrid = data.frame(k = seq(3, 51, 2)))
train_knn$bestTune

#b
plot(fit_knn)
  
# Harvard Answer
ggplot(train_knn)

#c
survived_hat <- predict(fit_knn, test_set) %>% factor(levels = levels(test_set$Survived))
cm_test <- confusionMatrix(data = survived_hat, reference = test_set$Survived)
cm_test$overall["Accuracy"]

# Harvard Answer
knn_preds <- predict(train_knn, test_set)
mean(knn_preds == test_set$Survived)

# Question 10 
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

# Harvard Answer
set.seed(8, sample.kind = "Rounding")    # simulate R 3.5
train_knn_cv <- train(Survived ~ .,
                      method = "knn",
                      data = train_set,
                      tuneGrid = data.frame(k = seq(3, 51, 2)),
                      trControl = trainControl(method = "cv", number = 10, p = 0.9))
train_knn_cv$bestTune

knn_cv_preds <- predict(train_knn_cv, test_set)
mean(knn_cv_preds == test_set$Survived)

# Question 11
#a
set.seed(10, sample.kind = 'Rounding')
fit_rpart11 <- train(Survived ~ ., 
                     data=train_set, 
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.05, 0.002)))
plot(fit_rpart11)
survived_hat <- predict(fit_rpart11, test_set)
cm_test <- confusionMatrix(data = survived_hat, reference = test_set$Survived)
cm_test$overall["Accuracy"]

#Harvard Answer
set.seed(10, sample.kind = "Rounding")    # simulate R 3.5
train_rpart <- train(Survived ~ ., 
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.05, 0.002)),
                     data = train_set)
train_rpart$bestTune

rpart_preds <- predict(train_rpart, test_set)
mean(rpart_preds == test_set$Survived)

#b
fit_rpart11$finalModel
plot(fit_rpart11$finalModel, margin=0.1)
text(fit_rpart11$finalModel, cex = 0.75)

# Harvard Answer
plot(train_rpart$finalModel, margin = 0.1)
text(train_rpart$finalModel)\

# Question 12

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

# Harvard Answer
# MATCHED!




