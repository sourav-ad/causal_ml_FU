###CAUSAL MACHINE LEARNING ASSIGNMENT###
#Sourav
#Eduard

#dependencies
library(MatchIt)
library(caret)
library(glmnet)

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
View(data_clean)
#nrow(data) =  17468
#ncol(data) = 16

#Task 1

#(a) Naive ATE estimator

#with treatment
y_1 <- data_clean$health1[data_clean$sportsclub == 1]
#without treatment
y_0 <- data_clean$health1[data_clean$sportsclub == 0]

ate_naive <- mean(y_1) - mean(y_0)
ate_naive # 0.1608723

#standard error, for ci
se_ate <- sqrt(var(y_1)/length(y_1) + var(y_0)/length(y_0))

#95% ci
ci <- c(
  ate_naive - 1.96 * se_ate,
  ate_naive + 1.96 * se_ate
)

ci # 0.1474207 0.1743239

#(b) probit model

controls <- c(
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

form_ps <- as.formula(
  paste("sportsclub ~", paste(controls, collapse = " + "))
)

ps_probit <- glm(
  form_ps,
  data   = data_clean,
  family = binomial(link = "probit")
)

data_clean$pscore <- predict(ps_probit, type = "response")

summary(ps_probit)
summary(data_clean$pscore)

#overlap check
range(data_clean$pscore[data_clean$sportsclub == 1])
range(data_clean$pscore[data_clean$sportsclub == 0])

# (c)

m_nn <- matchit(
  sportsclub ~ female + siblings + born_germany + parent_nongermany +
    newspaper + academictrack + urban + age + deutsch + bula +
    obese + eversmoked + currentsmoking + everalc,
  data     = data_clean,
  method   = "nearest",
  distance = data_clean$pscore,   # use propensity scores from probit model
  replace  = TRUE,
  estimand = "ATT"
)

matched_data <- match.data(m_nn)

att_nn <- with(
  matched_data,
  mean(health1[sportsclub == 1]) -
    mean(health1[sportsclub == 0])
)

att_nn

#propensity score weighting for the estimation of ATT

data_clean$w_att <- ifelse(
  data_clean$sportsclub == 1,
  1,
  data_clean$pscore / (1 - data_clean$pscore)
)

att_psw <- with(
  data_clean,
  weighted.mean(health1[sportsclub == 1], w_att[sportsclub == 1]) -
    weighted.mean(health1[sportsclub == 0], w_att[sportsclub == 0])
)

att_psw

att_model <- lm(
  health1 ~ sportsclub,
  data    = data_clean,
  weights = w_att
)

summary(att_model)

att_model <- lm(
  health1 ~ sportsclub,
  data    = data_clean,
  weights = w_att
)

summary(att_model)

summary(data_clean$w_att)

#(d)

#Problematic controls are those that violate the identification logic 
#of causal inference when conditioned on.

#TO BE PROPERLY EDITED

# Some control variables, such as obesity and smoking-related measures, 
# are likely affected by sports club membership and therefore constitute 
# post-treatment variables. 
# 
# Conditioning on them can bias the estimated treatment effect by 
# blocking causal pathways or inducing collider bias. 
# Other variables, such as academic track or regional fixed effects, 
# may strongly predict treatment while contributing little to outcome variation, 
# potentially worsening overlap and increasing variance. 
# 
# Propensity score methods rebalance observed covariates but 
# do not resolve bias arising from post-treatment conditioning 
# or poor covariate selection; 
# they only address imbalance in pre-treatment confounders.

#Task 2

#(a)

controls_clean <- c(
  "female",
  "siblings",
  "born_germany",
  "parent_nongermany",
  "newspaper",
  "academictrack",
  "urban",
  "age",
  "deutsch",
  "bula"
)

form_logit <- as.formula(
  paste("sportsclub ~", paste(controls_clean, collapse = " + "))
)

form_logit <- as.formula(
  paste("sportsclub ~", paste(controls_clean, collapse = " + "))
)

#cross validation 

ctrl_10 <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

logit_10 <- train(
  form_logit,
  data = data_clean,
  method = "glm",
  family = binomial(link = "logit"),
  trControl = ctrl_10,
  metric = "Accuracy"
)

logit_10


ctrl_5 <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

logit_5 <- train(
  form_logit,
  data = data_clean,
  method = "glm",
  family = binomial(link = "logit"),
  trControl = ctrl_5,
  metric = "Accuracy"
)

logit_5

cv_error_10 <- 1 - logit_10$results$Accuracy
cv_error_5  <- 1 - logit_5$results$Accuracy

cv_error_10
cv_error_5


#(b)

X <- model.matrix(
  as.formula(paste("sportsclub ~", paste(controls_clean, collapse = " + "))),
  data_clean
)[, -1]  # drop intercept

y <- data_clean$sportsclub

n <- nrow(X)
train_idx <- sample(seq_len(n), size = 0.7 * n)

X_train <- X[train_idx, ]
y_train <- y[train_idx]

X_test  <- X[-train_idx, ]
y_test  <- y[-train_idx]

cv_lasso <- cv.glmnet(
  X_train, y_train,
  family = "binomial",
  alpha  = 1,
  nfolds = 10
)

lambda_lasso <- cv_lasso$lambda.min

cv_ridge <- cv.glmnet(
  X_train, y_train,
  family = "binomial",
  alpha  = 0,
  nfolds = 10
)

lambda_ridge <- cv_ridge$lambda.min

cv_elnet <- cv.glmnet(
  X_train, y_train,
  family = "binomial",
  alpha  = 0.5,
  nfolds = 10
)

lambda_elnet <- cv_elnet$lambda.min

fit_lasso <- glmnet(
  X_train, y_train,
  family = "binomial",
  alpha  = 1,
  lambda = lambda_lasso
)

fit_ridge <- glmnet(
  X_train, y_train,
  family = "binomial",
  alpha  = 0,
  lambda = lambda_ridge
)

fit_elnet <- glmnet(
  X_train, y_train,
  family = "binomial",
  alpha  = 0.5,
  lambda = lambda_elnet
)
