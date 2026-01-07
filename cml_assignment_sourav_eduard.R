###CAUSAL MACHINE LEARNING ASSIGNMENT###
#Sourav Adhikari
#Eduard-Alex Ciuhandu

#dependencies
packages <- c(
  "tidyverse",
  "boot",
  "MatchIt",
  "caret",
  "glmnet",
  "glmnetUtils",
  "tree",
  "randomForest",
  "gbm"
)

installed <- rownames(installed.packages())

for (p in packages) {
  if (!p %in% installed) {
    install.packages(p, dependencies = TRUE)
  }
  library(p, character.only = TRUE)
}

set.seed(123)

#Load data
data <- read.csv("data/assignment_data.csv")
#nrow(data) = 19850
#ncol(data) = 158

#Take subset with following columns
vars <- c(
  "health1", "sportsclub", "female", "siblings", "born_germany",
  "parent_nongermany", "newspaper", "academictrack", "urban",
  "age", "deutsch", "bula", "obese", "eversmoked",
  "currentsmoking", "everalc"
)

#Outcome = health1
#Treatment = sportsclub

data <- data[, vars]

#preprocessing: remove all NA containing rows, if at least one entry is NA
data_clean <- na.omit(data)
#View(data_clean)
#nrow(data_clean) =  17468
#ncol(data_clean) = 16

#read "bula" as factor
data_clean$bula <- factor(data_clean$bula)

## Task 1 ####

### (a) #####
#Compute the naive estimator for the average treatment effect (ATE) 
#and the 95% confidence interval (you can assume a normal distribution). 

#with treatment
y_1 <- data_clean$health1[data_clean$sportsclub == 1]
#without treatment
y_0 <- data_clean$health1[data_clean$sportsclub == 0]

ate_naive <- mean(y_1) - mean(y_0)
ate_naive # 0.1608723

#standard error, for ci
se_ate <- sqrt(var(y_1)/length(y_1) + var(y_0)/length(y_0))

#95% confidence interval 
ci <- c(
  ate_naive - 1.96 * se_ate,
  ate_naive + 1.96 * se_ate
)

ci # 0.1474207 0.1743239

### (b) Estimate the probability of treatment using all control variables (probit). ####

controls <- c( #exclude outcome and treatment
  "female",
  "siblings",
  "born_germany",
  "parent_nongermany",
  "newspaper",
  "academictrack",
  "urban",
  "age",
  "deutsch",
  "bula",
  "obese",
  "eversmoked",
  "currentsmoking",
  "everalc"
)

#formula
form_ps <- as.formula(
  paste("sportsclub ~", paste(controls, collapse = " + "))
)

#probit estimation
ps_probit <- glm(
  form_ps,
  data   = data_clean,
  family = binomial(link = "probit")
)

#predicted propensity scores
data_clean$pscore <- predict(ps_probit, type = "response")

summary(ps_probit)

summary(data_clean$pscore)
#estimated probability of treatment
#OUTPUT
#  Min.   1st Qu.  Median   Mean   3rd Qu.  Max. 
# 0.01489 0.33682 0.43410 0.42976 0.52423 0.99996  

#overlap check
range(data_clean$pscore[data_clean$sportsclub == 1])
# 0.05269266 0.99995669

range(data_clean$pscore[data_clean$sportsclub == 0])
# 0.01488587 0.78271074

### (c) Estimate the Average Treatment Effect on the Treated (ATT) ####
# using nearest-neighbor matching with replacement.

#nearest-neighbor matching on propensity score (with replacement)
#manual implementation

treated  <- data_clean[data_clean$sportsclub == 1, ]
control  <- data_clean[data_clean$sportsclub == 0, ]

matches <- sapply(treated$pscore, function(p) {
  which.min(abs(control$pscore - p))
})

y_treated  <- treated$health1
y_matched  <- control$health1[matches]

att_nn <- mean(y_treated - y_matched)

att_nn
# 0.09766822

# using propensity score weighting for the estimation of ATT

# reweights controls to resemble the treated group

#ATT weights
data_clean$w_att <- ifelse(
  data_clean$sportsclub == 1,
  1,
  data_clean$pscore / (1 - data_clean$pscore)
)

#using weighted means
att_psw <- with(
  data_clean,
  weighted.mean(health1[sportsclub == 1], w_att[sportsclub == 1]) -
    weighted.mean(health1[sportsclub == 0], w_att[sportsclub == 0])
)

att_psw # 0.1422038


# regression formulation
#sanity check
att_model <- lm(
  health1 ~ sportsclub,
  data    = data_clean,
  weights = w_att
)

summary(att_model)

# coef(att_model)["sportsclub"] matches att_psw 

summary(data_clean$w_att)

### (d) #### 

# Variables such as obesity, smoking, and alcohol consumption are most likely affected
# by having sports club membership and therefore would constitute post-treatment variables.
# Demographic and background variables are
# determined prior to treatment and can be considered unproblematic controls.
# Propensity score methods do not solve the problem of post-treatment variables:
# including such variables in the propensity score model would still bias the
# estimated treatment effect, since the issue is causal ordering, not imbalance.

#Thus, the non problematic variables are as follows:

controls_unproblematic <- c(
  "female",
  "age",
  "siblings",
  "born_germany",
  "parent_nongermany",
  "deutsch",
  "urban",
  "academictrack",
  "newspaper",
  "bula"
)

## Task 2 ####

### (a) #####
# Compute and compare the 10-fold and 5-fold cross-validation errors resulting 
# from fitting a logistic regression model 
# with control variables deemed unproblematic in 1d).


# outcome is for classification
data_clean$health1_f <- factor(
  data_clean$health1,
  levels = c(0, 1),
  labels = c("No", "Yes")
)

#logistic regression, still conditions on the treatment, additionally on
#unproblematic covariates
form_logit <- as.formula(
  paste("health1_f ~ sportsclub +",  #health1 as yes/no
        paste(controls_unproblematic, collapse = " + "))
)

#cross validation 

#10 fold
ctrl_10 <- trainControl(
  method = "cv",
  number = 10
)

logit_10 <- train(
  form_logit,
  data = data_clean,
  method = "glm",
  family = binomial,
  trControl = ctrl_10
)

logit_10 #warning arises because Bundesland is included as a factor

#  Accuracy   Kappa    
# 0.7267574  -0.001367787


#5 fold

ctrl_5 <- trainControl(
  method = "cv",
  number = 5
)

logit_5 <- train(
  form_logit,
  data = data_clean,
  method = "glm",
  family = binomial,
  trControl = ctrl_5
)

logit_5

#  Accuracy   Kappa    
# 0.7265859  -0.001326337

cv_10_error <- 1 - logit_10$results$Accuracy
cv_5_error  <- 1 - logit_5$results$Accuracy

cv_10_error # 0.2732426
cv_5_error # 0.2734141


### (b) ####
#Split the data into a 70% training and 30% test set. 
#Estimate lasso, ridge, and elastic net models using 
#cross-validation to choose penalty parameters. 


# 70-30 train test split

n <- nrow(data_clean)
train_idx <- sample(seq_len(n), size = 0.7 * n)

data_train <- data_clean[train_idx, ]
data_test  <- data_clean[-train_idx, ]

#sanity check
nrow(data_train) / nrow(data_clean)  # 0.6999657
nrow(data_test)  / nrow(data_clean)  # 0.3000343

X_train <- model.matrix(
  as.formula(paste("health1 ~ sportsclub +", paste(controls_unproblematic, collapse = " + "))),
  data_train
)[, -1]   # drop intercept

y_train <- data_train$health1

#Using 5-fold cv

#lasso, alpha = 1
cv_lasso <- cv.glmnet(
  X_train,
  y_train,
  family = "binomial",
  alpha  = 1,
  nfolds = 5
)

#the best lambda parameter
lambda_lasso <- cv_lasso$lambda.min
lambda_lasso # 0.002400034

#final model with optimal lambda
lasso_model <- glmnet(
  X_train,
  y_train,
  family = "binomial",
  alpha  = 1,
  lambda = lambda_lasso
)

#Ridge, alpha = 0
#same X_train and y_train

cv_ridge <- cv.glmnet(
  X_train,
  y_train,
  family = "binomial",
  alpha  = 0,
  nfolds = 5
)

lambda_ridge <- cv_ridge$lambda.min
lambda_ridge # 0.02759455

#final model

ridge_model <- glmnet(
  X_train,
  y_train,
  family = "binomial",
  alpha  = 0,
  lambda = lambda_ridge
)

#Elastic Net, alpha = 0.5
#again same X_train and y_train

cv_elastic <- cv.glmnet(
  X_train,
  y_train,
  family = "binomial",
  alpha  = 0.5,
  nfolds = 5
)

lambda_elastic <- cv_elastic$lambda.min
lambda_elastic # 0.005781698

#final model with best lambda
elastic_model <- glmnet(
  X_train,
  y_train,
  family = "binomial",
  alpha  = 0.5,
  lambda = lambda_elastic
)

# Grid search for alpha and lambda:
#set.seed(123)

# cv_elastic_grid <- cva.glmnet(
#   X_train,
#   y_train,
#   alpha = seq(0, 1, 1/100),
#   family = "binomial",
#   nfolds = 5
# )
# 
# min_cvm <- sapply(cv_elastic_grid$modlist, function(m) min(m$cvm))
# best_i <- which.min(min_cvm)
# 
# alpha_elastic_grid <- cv_elastic_grid$alpha[best_i]
# lambda_elastic_grid <- cv_elastic_grid$modlist[[best_i]]$lambda.min
# 
# alpha_elastic_grid # 1 - basically lasso
# lambda_elastic_grid #  0.001394045 - exactly lambda_lasso
# 
# range(min_cvm)
# 1.136796 1.136904
# Very close range, suggests that all alphas perform very similarly

### (c) #####
#Evaluate all three models on the test set and compare their prediction errors. ###

#test data

X_test <- model.matrix(
  as.formula(
    paste(
      "health1 ~ sportsclub +",
      paste(controls_unproblematic, collapse = " + ")
    )
  ),
  data_test
)[, -1]

y_test <- data_test$health1

#make predictions, predicted probabilities
p_lasso <- predict(cv_lasso, newx = X_test, s = "lambda.min", type = "response")
p_ridge <- predict(cv_ridge, newx = X_test, s = "lambda.min", type = "response")
p_elastic <- predict(cv_elastic, newx = X_test, s = "lambda.min", alpha = 0.5, type = "response")

#convert to class predictions
yhat_lasso <- ifelse(p_lasso > 0.5, 1, 0)
yhat_ridge <- ifelse(p_ridge > 0.5, 1, 0)
yhat_elastic <- ifelse(p_elastic > 0.5, 1, 0)

#prediction errors
err_lasso <- mean(yhat_lasso != y_test)
err_ridge <- mean(yhat_ridge != y_test)
err_elastic <- mean(yhat_elastic != y_test)

err_lasso # 0.2703683
err_ridge # 0.2701774
err_elastic # 0.2701774

#sanity check
cor(p_lasso, p_ridge)
#            lambda.min
# lambda.min  0.9948542

cor(p_lasso, p_elastic)
#            lambda.min
# lambda.min  0.999769

### (d) #####
#Compare the coefficients across the three penalized models and ###
#comment briefly on differences.

#coefficients
coef_lasso <- coef(cv_lasso, s = "lambda.min")
coef_ridge <- coef(cv_ridge, s = "lambda.min")
coef_elastic <- coef(cv_elastic, alpha = 0.5, s = "lambda.min")



coef_to_df <- function(coef_obj, model_name) {
  df <- as.data.frame(as.matrix(coef_obj))
  colnames(df) <- model_name
  df$term <- rownames(df)
  rownames(df) <- NULL
  df
}

df_lasso <- coef_to_df(coef_lasso, "LASSO")
df_ridge <- coef_to_df(coef_ridge, "Ridge")
df_elastic <- coef_to_df(coef_elastic, "ElasticNet")

coef_compare <- Reduce(
  function(x, y) merge(x, y, by = "term", all = TRUE),
  list(df_lasso, df_ridge, df_elastic)
)

coef_compare

#               term       LASSO         Ridge    ElasticNet
# 1        (Intercept) -0.02476756  0.06975852 -0.073334643
# 2      academictrack  0.12251166  0.13947400  0.118473906
# 3                age -0.05094313 -0.05324894 -0.049251201
# 4       born_germany  0.00000000 -0.02829949  0.000000000
# 5             bula10  0.00000000  0.23522951  0.000000000
# 6             bula11  0.40583749  0.86300986  0.270853684
# 7             bula12  0.67888810  0.99787150  0.560588781
# 8             bula13 -0.03142268 -0.07995142 -0.033048144
# 9             bula14  0.36486661  0.54431984  0.289898642
# 10            bula15  0.00000000  0.26041922  0.000000000
# 11            bula16  0.00000000 -0.04730675  0.000000000
# 12             bula2  0.31831882  0.53522407  0.222493877
# 13             bula3  0.18618509  0.25273854  0.147920983
# 14             bula4  0.12178935  0.09524748  0.112928748
# 15             bula5 -0.19872112 -1.98972017  0.000000000
# 16             bula6  0.00000000  0.46318182  0.000000000
# 17             bula7  0.00000000 -0.21211868  0.000000000
# 18             bula8 -1.16739870 -1.85535855 -0.938337556
# 19             bula9  0.00000000 -0.55546188  0.000000000
# 20           deutsch -0.52576701 -0.52044935 -0.488137839
# 21            female -0.06854832 -0.08709702 -0.063118304
# 22         newspaper  0.04857751  0.06825769  0.044589211
# 23 parent_nongermany -0.12116368 -0.15050516 -0.101882986
# 24          siblings  0.01292005  0.04492458  0.005381157
# 25        sportsclub  0.75457460  0.67698086  0.740808071
# 26             urban  0.00000000  0.02039875  0.000000000

#Interpretation

# The coefficients inferred are consistent across L1, L2
# and elastic net regularization. Lasso and Elastic net shrink
# "born_germany" and "urban" to 0 while ridge does not. This indicates
# limited predictive contribution for both of them. The coefficient on 
# "sportsclub" is large, positive, and stable across all three models, 
# indicating a strong association with good health. Overall, the 3 models
# show similar predictive performance.


## Task 3 #####

### (a) #####

data_clean_tree <- data_clean[, c("health1", "sportsclub", controls_unproblematic)]


data_clean_tree$health1 <- as.factor(data_clean_tree$health1)

classTree <- tree(formula = health1 ~ ., 
                  data = data_clean_tree, 
                  subset = train_idx, 
                  split = "gini")

summary(classTree)

# The tree has 544 terminal nodes, average impurity after splitting of 1.126, 
# and around 25.96% of the training observations are misclassified.

### (b) #####

p <- ncol(data_clean_tree) - 1

set.seed(123)
bag <- randomForest(formula = health1 ~ ., 
                    ntree = 500, 
                    importance = TRUE, 
                    mtry = p, 
                    data = data_clean_tree, 
                    subset = train_idx)

bag_correct <- mean(predict(bag, 
                            newdata = data_clean_tree[-train_idx,]) == data_clean_tree[-train_idx, "health1"])
bag_miss <- 1 - bag_correct 

bag_miss # 0.3024232 - ~ 30%

set.seed(123)
rF <- randomForest(formula = health1 ~ ., 
                   ntree = 500, 
                   importance = TRUE, 
                   data = data_clean_tree, 
                   subset = train_idx)

rF_correct <- mean(predict(rF, 
                           newdata = data_clean_tree[-train_idx,]) == data_clean_tree[-train_idx, "health1"])
rF_miss <- 1 - rF_correct

rF_miss # 0.2739935 - ~ 27%

# The random forest has a lower misclassification rate than bagging (by about 2.8 pp)

### (c) #####
data_clean_tree$health1 <- as.numeric(as.character(data_clean_tree$health1))

set.seed(123)
boost <- gbm(formula = health1 ~ ., 
             distribution = "bernoulli", 
             data = data_clean_tree[train_idx,], n.trees = 1000)

boost_pred_test <- ifelse(predict(boost, 
                                  data_clean_tree[-train_idx,], 
                                  n.trees = 1000, 
                                  type = "response") > 0.5, 1, 0)

boost_correct <- mean(boost_pred_test == data_clean_tree[-train_idx, "health1"])
boost_miss <- 1 - boost_correct
boost_miss #0.2703683

### (d) #####

summary(boost)

# According to the relative influence measures from the boosted model, Bundesland and sport's club membership are 
# the most important predictors, followed by age, German citizenship, and academic track.