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
#View(data_clean)
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

#95% confidence interval 
ci <- c(
  ate_naive - 1.96 * se_ate,
  ate_naive + 1.96 * se_ate
)

ci # 0.1474207 0.1743239

#(b) Estimate the probability of treatment using all control variables (probit).

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
#  Min.     1st Qu.   Median    Mean    3rd Qu.    Max. 
# 0.01483  0.33720  0.43420   0.42977 0.52316   0.78342 

#overlap check
range(data_clean$pscore[data_clean$sportsclub == 1])
# 0.03424576 0.72989473

range(data_clean$pscore[data_clean$sportsclub == 0])
# 0.01483453 0.78342370

# (c) Estimate the Average Treatment Effect on the Treated (ATT) using
# nearest-neighbor matching with replacement.

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
# 0.1041972

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

att_psw # 0.1421826


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

#(d) 

# Variables such as obesity, smoking, and alcohol consumption are most likely affected
# by having sports club membership and therefore would constitute post-treatment variables.
# Conditioning on them may bias the estimated treatment effect by blocking causal
# pathways or inducing collider bias. Demographic and background variables are
# determined prior to treatment and can be considered unproblematic controls.
# Propensity score methods rebalance observed covariates but do not correct bias
# arising from conditioning on post-treatment variables.

#Task 2

#(a) Compute and compare the 10-fold and 5-fold cross-validation errors resulting 
# from fitting a logistic regression model 
# with control variables deemed unproblematic in 1d).

data_clean$sportsclub_f <- factor(
  data_clean$sportsclub,
  levels = c(0, 1),
  labels = c("No", "Yes")
)



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
  paste("sportsclub_f ~", paste(controls_clean, collapse = " + "))
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


#From Eduard

## Task 3 #####

X_full <- bind_cols(X, y)
colnames(X_full)[11] <- "health1"
X_full <- X_full %>%
  mutate(
    female = as.factor(female),
    siblings = as.factor(siblings),
    born_germany = as.factor(born_germany),
    parent_nongermany = as.factor(parent_nongermany),
    newspaper = as.factor(newspaper),
    academictrack = as.factor(academictrack),
    urban = as.factor(urban),
    deutsch = as.factor(deutsch),
    bula = as.character(bula),
    health1 = as.factor(health1)
  )

classTree <- tree(formula = health1 ~ ., data = X_full, subset = train_idx, split = "gini")

set.seed(123)
classCv <- cv.tree(object = classTree, FUN = prune.misclass)
treePrune <- prune.misclass(tree = classTree, best = 3)

plot(treePrune)
text(treePrune, pretty=0)
