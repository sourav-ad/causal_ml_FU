###CAUSAL MACHINE LEARNING ASSIGNMENT###
#Sourav
#Eduard

#dependencies
library(MatchIt)

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
