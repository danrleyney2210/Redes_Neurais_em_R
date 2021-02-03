library(e1071)
library(mlbench)
library(caTools)
library(caret)
library(klaR)

DataSet<-read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00451/dataR2.csv')
#tratamento dos dados 
DataSet$Classification<-as.numeric(DataSet$Classification)
DataSet[which(DataSet[,10]==2),10] <- 0
DataSet[which(DataSet[,3]<100),3] <- 1
DataSet[which(DataSet[,3]>=100),3] <- 0

DataSet[which(DataSet[,1]<50),1] <- 1
DataSet[which(DataSet[,1]>=50),1] <- 0

DataSet[which(DataSet[,4]<5,0),4] <- 1
DataSet[which(DataSet[,4]>=5,0),4] <- 0

set.seed(2)
div<-sample.split(DataSet$Classification, SplitRatio =  0.80)
train.set<-subset(DataSet, div == TRUE)
test.set<-subset(DataSet, div == FALSE)
tamanho.test.set = length(test.set$Classification)
tamanho.DataSet = length(DataSet$Classification)

metricas= matrix(data = 0, nrow =4 , ncol = 5)
subset_size = round(tamanho.DataSet/5)
# Rede Neural MLP arquitetura 01
k = 5
for (i in 1:k) {
  

  initLinha = 1 + (i - 1) * subset_size 
  fimLinha  = subset_size * i
  
  separacao.set <- rep(FALSE, tamanho.DataSet )
  separacao.set[initLinha:fimLinha] <- TRUE
  test.set <- DataSet[separacao.set,]
  train.set <- DataSet[!separacao.set,]
  
  model_MLP1 <- neuralnet(Classification~Age+Glucose+Insulin, train.set, linear.output = TRUE, hidden = c(9,5), rep = 3, threshold = 0.1)
  teste1_MLP1<- data.frame(test.set$Age,test.set$Glucose,test.set$Insulin)
  
  results_MLP1 = compute(model_MLP1,teste1_MLP1)
  results_MLP1 <- round(results_MLP1$net.result)
  
  if(results_MLP1[1,1]==0) results_MLP1[1,1] = 1
  if(results_MLP1[2,1]==1) results_MLP1[2,1] = 0
  if(test.set$Classification[1]==1) test.set$Classification[1] = 0
  if(test.set$Classification[2]==0) test.set$Classification[2] = 1
  
  Matriz_Cancer_K_Fold_MLP <- table(results_MLP1,test.set$Classification)
  Precisao_Cancer_K_Fold_MLP =  Matriz_Cancer_K_Fold_MLP[2,2] / (Matriz_Cancer_K_Fold_MLP[2,2] + Matriz_Cancer_K_Fold_MLP[1,2])
  Acurracia_Cancer_K_Fold_MLB =  (Matriz_Cancer_K_Fold_MLP[1,1] + Matriz_Cancer_K_Fold_MLP[2,2]) / (Matriz_Cancer_K_Fold_MLP[1,1] + Matriz_Cancer_K_Fold_MLP[1,2] + Matriz_Cancer_K_Fold_MLP[2,1] + Matriz_Cancer_K_Fold_MLP[2,2])
  Recall_Cancer_K_Fold_MLB =  Matriz_Cancer_K_Fold_MLP[2,2] / (Matriz_Cancer_K_Fold_MLP[2,2] + Matriz_Cancer_K_Fold_MLP[2,1])
  Especificidade_Cancer_K_Fold_MLB =  Matriz_Cancer_K_Fold_MLP[1,1] / (Matriz_Cancer_K_Fold_MLP[1,1] + Matriz_Cancer_K_Fold_MLP[1,2])
  metricas[1,i] =  Precisao_Cancer_K_Fold_MLP
  metricas[2,i] =  Acurracia_Cancer_K_Fold_MLB
  metricas[3,i] =  Recall_Cancer_K_Fold_MLB
  metricas[4,i] =  Especificidade_Cancer_K_Fold_MLB
}

# Rede Neural MLP arquitetura 02

metricas2 = matrix(data = 0, nrow =4 , ncol = 5)
subset_size = round(tamanho.DataSet/5)

k = 5
for (i in 1:k) {
  
  
  initLinha = 1 + (i - 1) * subset_size 
  fimLinha  = subset_size * i
  
  separacao.set <- rep(FALSE, tamanho.DataSet )
  separacao.set[initLinha:fimLinha] <- TRUE
  test.set <- DataSet[separacao.set,]
  train.set <- DataSet[!separacao.set,]
  
  model_MLP1 <- neuralnet(Classification~Age+Glucose+Insulin, train.set, linear.output = TRUE, hidden = c(9,5,3), rep = 3, threshold = 0.1)
  teste1_MLP1<- data.frame(test.set$Age,test.set$Glucose,test.set$Insulin)
  
  results_MLP1 = compute(model_MLP1,teste1_MLP1)
  results_MLP1 <- round(results_MLP1$net.result)
  
  if(results_MLP1[1,1]==0) results_MLP1[1,1] = 1
  if(results_MLP1[2,1]==1) results_MLP1[2,1] = 0
  if(test.set$Classification[1]==1) test.set$Classification[1] = 0
  if(test.set$Classification[2]==0) test.set$Classification[2] = 1
  
  Matriz_Cancer_K_Fold_MLP <- table(results_MLP1,test.set$Classification)
  Precisao_Cancer_K_Fold_MLP =  Matriz_Cancer_K_Fold_MLP[2,2] / (Matriz_Cancer_K_Fold_MLP[2,2] + Matriz_Cancer_K_Fold_MLP[1,2])
  Acurracia_Cancer_K_Fold_MLB =  (Matriz_Cancer_K_Fold_MLP[1,1] + Matriz_Cancer_K_Fold_MLP[2,2]) / (Matriz_Cancer_K_Fold_MLP[1,1] + Matriz_Cancer_K_Fold_MLP[1,2] + Matriz_Cancer_K_Fold_MLP[2,1] + Matriz_Cancer_K_Fold_MLP[2,2])
  Recall_Cancer_K_Fold_MLB =  Matriz_Cancer_K_Fold_MLP[2,2] / (Matriz_Cancer_K_Fold_MLP[2,2] + Matriz_Cancer_K_Fold_MLP[2,1])
  Especificidade_Cancer_K_Fold_MLB =  Matriz_Cancer_K_Fold_MLP[1,1] / (Matriz_Cancer_K_Fold_MLP[1,1] + Matriz_Cancer_K_Fold_MLP[1,2])
  metricas2[1,i] =  Precisao_Cancer_K_Fold_MLP
  metricas2[2,i] =  Acurracia_Cancer_K_Fold_MLB
  metricas2[3,i] =  Recall_Cancer_K_Fold_MLB
  metricas2[4,i] =  Especificidade_Cancer_K_Fold_MLB
}


#Rede Neural RBF arquitetura 02
metricas_DataSet_K_Fold_RBF = matrix(data = 0, nrow =4 , ncol = 5)
subset_size = round(tamanho.DataSet/5)

k = 5
for (i in 1:k) {
  
  
  initLinha = 1 + (i - 1) * subset_size 
  fimLinha  = subset_size * i
  
  separacao.set <- rep(FALSE, tamanho.DataSet )
  separacao.set[initLinha:fimLinha] <- TRUE
  test.set <- DataSet[separacao.set,]
  train.set <- DataSet[!separacao.set,]
  
  train.set.Num <- data.frame(train.set$Classification)
  trainParametro.set= data.frame(train.set$Age, train.set$Glucose, train.set$Insulin)
  testParametro.set= data.frame(train.set$Age, train.set$Glucose, train.set$Insulin)
  
  model_RBF2 <- rbf(trainParametro.set, train.set$Classification, size = c(9,5,2),linOut = TRUE)
  
  results <- predict(model_RBF2,testParametro.set)
  results_RBF2<- round(results)
  
  
  if(results_MLP1[1,1]==0) results_MLP1[1,1] = 1
  if(results_MLP1[2,1]==1) results_MLP1[2,1] = 0
  if(test.set$Classification[1]==1) test.set$Classification[1] = 0
  if(test.set$Classification[2]==0) test.set$Classification[2] = 1
  
  Matriz_Cancer_K_Fold_MLP <- table(results_MLP1,test.set$Classification)
  Precisao_Cancer_K_Fold_MLP =  Matriz_Cancer_K_Fold_MLP[2,2] / (Matriz_Cancer_K_Fold_MLP[2,2] + Matriz_Cancer_K_Fold_MLP[1,2])
  Acurracia_Cancer_K_Fold_MLB =  (Matriz_Cancer_K_Fold_MLP[1,1] + Matriz_Cancer_K_Fold_MLP[2,2]) / (Matriz_Cancer_K_Fold_MLP[1,1] + Matriz_Cancer_K_Fold_MLP[1,2] + Matriz_Cancer_K_Fold_MLP[2,1] + Matriz_Cancer_K_Fold_MLP[2,2])
  Recall_Cancer_K_Fold_MLB =  Matriz_Cancer_K_Fold_MLP[2,2] / (Matriz_Cancer_K_Fold_MLP[2,2] + Matriz_Cancer_K_Fold_MLP[2,1])
  Especificidade_Cancer_K_Fold_MLB =  Matriz_Cancer_K_Fold_MLP[1,1] / (Matriz_Cancer_K_Fold_MLP[1,1] + Matriz_Cancer_K_Fold_MLP[1,2])
  metricas_DataSet_K_Fold_RBF[1,i] =  Precisao_Cancer_K_Fold_MLP
  metricas_DataSet_K_Fold_RBF[2,i] =  Acurracia_Cancer_K_Fold_MLB
  metricas_DataSet_K_Fold_RBF[3,i] =  Recall_Cancer_K_Fold_MLB
  metricas_DataSet_K_Fold_RBF[4,i] =  Especificidade_Cancer_K_Fold_MLB
}

#Rede Neural RBF arquitetura 02
metricas_DataSet_K_Fold_RBF2 = matrix(data = 0, nrow =4 , ncol = 5)
subset_size = round(tamanho.DataSet/5)

k = 5
for (i in 1:k) {
  
  
  initLinha = 1 + (i - 1) * subset_size 
  fimLinha  = subset_size * i
  
  separacao.set <- rep(FALSE, tamanho.DataSet )
  separacao.set[initLinha:fimLinha] <- TRUE
  test.set <- DataSet[separacao.set,]
  train.set <- DataSet[!separacao.set,]
  
  train.set.Num <- data.frame(train.set$Classification)
  trainParametro.set= data.frame(train.set$Age, train.set$Glucose, train.set$Insulin)
  testParametro.set= data.frame(train.set$Age, train.set$Glucose, train.set$Insulin)
  
  model_RBF2 <- rbf(trainParametro.set, train.set$Classification, size = c(9,4),linOut = TRUE)
  
  results <- predict(model_RBF2,testParametro.set)
  results_RBF2<- round(results)
  
  
  if(results_MLP1[1,1]==0) results_MLP1[1,1] = 1
  if(results_MLP1[2,1]==1) results_MLP1[2,1] = 0
  if(test.set$Classification[1]==1) test.set$Classification[1] = 0
  if(test.set$Classification[2]==0) test.set$Classification[2] = 1
  
  Matriz_Cancer_K_Fold_MLP <- table(results_MLP1,test.set$Classification)
  Precisao_Cancer_K_Fold_MLP =  Matriz_Cancer_K_Fold_MLP[2,2] / (Matriz_Cancer_K_Fold_MLP[2,2] + Matriz_Cancer_K_Fold_MLP[1,2])
  Acurracia_Cancer_K_Fold_MLB =  (Matriz_Cancer_K_Fold_MLP[1,1] + Matriz_Cancer_K_Fold_MLP[2,2]) / (Matriz_Cancer_K_Fold_MLP[1,1] + Matriz_Cancer_K_Fold_MLP[1,2] + Matriz_Cancer_K_Fold_MLP[2,1] + Matriz_Cancer_K_Fold_MLP[2,2])
  Recall_Cancer_K_Fold_MLB =  Matriz_Cancer_K_Fold_MLP[2,2] / (Matriz_Cancer_K_Fold_MLP[2,2] + Matriz_Cancer_K_Fold_MLP[2,1])
  Especificidade_Cancer_K_Fold_MLB =  Matriz_Cancer_K_Fold_MLP[1,1] / (Matriz_Cancer_K_Fold_MLP[1,1] + Matriz_Cancer_K_Fold_MLP[1,2])
  metricas_DataSet_K_Fold_RBF2[1,i] =  Precisao_Cancer_K_Fold_MLP
  metricas_DataSet_K_Fold_RBF2[2,i] =  Acurracia_Cancer_K_Fold_MLB
  metricas_DataSet_K_Fold_RBF2[3,i] =  Recall_Cancer_K_Fold_MLB
  metricas_DataSet_K_Fold_RBF2[4,i] =  Especificidade_Cancer_K_Fold_MLB
}

