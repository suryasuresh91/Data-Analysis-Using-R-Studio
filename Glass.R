mydata <- read.table("glass.csv", sep=",", header=TRUE)
head(mydata)
summary(mydata)

#Print Histogram
hist(mydata$glasstype)
summary(mydata$glasstype)
table(mydata$glasstype)

# Fit a linear model
mylm <- lm(formula = glasstype ~ id + RI + Na + Mg + Al + Si + K + Ca + Ba + Fe, data = mydata)
summary(mylm)


#Assigning class groups
mydata$windowglass <- mydata$glasstype <=4
mydata$nonwindowglass <- mydata$glasstype >=5

#Fits the model using logistic regression
#Considering only significant predictors
log2_good = glm(class1 ~ id + Al,family=binomial(link="logit"),data = mydata)
summary(log2_good)
 
#-------------------------------------------------------------------------------
#---------------------------Empirical evaluation--------------------------------

myresponse <- factor(mydata[,14])
myresponse
mydf <- data.frame(myresponse, mydata[,1:11])
mydf
numobs <- nrow(mydf)
numobs

# Part I: k-fold cross validation (classification tree)
set.seed(1)
numFolds <- 10

# Assign observations to k groups
xvalFoldNumber <- sample(1:numobs %% numFolds + 1,replace=FALSE)
xvalFoldNumber

# Create a list of test observations for each group
xvalSets <- lapply(1:numFolds, FUN=function(x) {
  list(test=which(xvalFoldNumber == x))
})
xvalSets

library(rpart)
# Create a function for each group
rpartFold <- function(x) {
  testdf <- mydf[x$test,]
  traindf <- mydf[-x$test,]
  
  myrpart <- rpart(myresponse ~ ., data=traindf)
  ## classification predictions
  myrpartPredict <- predict(myrpart, newdata=testdf, type="class")
  confusion <- table(testdf[,1], myrpartPredict) 
  confusion
}

# Apply the function to each group
myrpartResults <- lapply(xvalSets, FUN=rpartFold)
myrpartResults

# Sum up all the results
totalConfusion <- Reduce("+", myrpartResults)
totalConfusion
totalConfusion/rowSums(totalConfusion)

#-------------------------------------------------------------------------------
#------------------------------------PCA----------------------------------------

library(rgl)
mydata <- mydata[,1:11]
mydata


mydata <- scale(mydata, scale = FALSE)

mypca <- prcomp(mydata, retx=TRUE)
mypca$rotation # rotation/loadings matrix
mypca$x  # scores

# Determine the number of PCs
summary(mypca)
mypca$sdev
myvar <- mypca$sdev^2 # variance explained by each PC
myvar/sum(myvar) # % proportion of variance explained by each PC
barplot(myvar)

plot(mypca) # scree plot
mypca$PC1
plot(summary(mypca)$importance[3,])
pcs <- as.data.frame(mypca$x)
pcs
mydfpca <- data.frame(myresponse, 
                      pcs[,1:5])
mydfpca
numobspca <- nrow(mydfpca)
numobspca
log_good_pca = glm(myresponse~PC1 + PC2 + PC3 + PC4 + PC5,data=mydfpca,family=binomial(link="logit"))


summary(log_good_pca)

# -----------------------------------------------------------------------------
#---------------Empirical evaluation for Principal components------------------


# Part I: k-fold cross validation (classification tree)
set.seed(1)
numFolds <- 10

# Assign observations to k groups
xvalFoldNumber <- sample(1:numobspca %% numFolds + 1,
                         replace=FALSE)
xvalFoldNumber

# Create a list of test observations for each group
xvalSets <- lapply(1:numFolds, FUN=function(x) {
  list(test=which(xvalFoldNumber == x))
})
xvalSets

library(rpart)
# Create a function for each group
rpartFold <- function(x) {
  testdf <- mydfpca[x$test,]
  traindf <- mydfpca[-x$test,]
  
  myrpart <- rpart(myresponse ~ ., data=traindf)
  ## classification predictions
  myrpartPredict <- predict(myrpart, newdata=testdf, type="class")
  confusion <- table(testdf[,1], myrpartPredict) 
  confusion
}

# Apply the function to each group
myrpartResults <- lapply(xvalSets, FUN=rpartFold)
myrpartResults

# Sum up all the results
totalConfusion <- Reduce("+", myrpartResults)
totalConfusion
totalConfusion/rowSums(totalConfusion) 

