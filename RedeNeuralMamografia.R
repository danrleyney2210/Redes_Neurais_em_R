library(e1071)
library(mlbench)
library(caTools)
library(caret)
library(klaR)



DataSet<-read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data')
colnames(DataSet)=c('avaliation','age','form','marg','densidade','gravidade')

DataSet[which(DataSet[,1]== '?'),1] <- '1'
DataSet[which(DataSet[,2]== '?'),2] <- '1'
DataSet[which(DataSet[,3]== '?'),3] <- '1'
DataSet[which(DataSet[,4]== '?'),4] <- '1'
DataSet[which(DataSet[,5]== '?'),5] <- '1'

DataSet[which(DataSet[,3]==1),3] <- 0
DataSet[which(DataSet[,3]>1),3] <- 1

DataSet$avaliation<-as.numeric(DataSet$avaliation)
DataSet$age<-as.numeric(DataSet$age)
DataSet$form<-as.numeric(DataSet$form)
DataSet$marg<-as.numeric(DataSet$marg)
DataSet$densidade<-as.numeric(DataSet$densidade)

set.seed(2)
DataSet$gravidade =  as.numeric(DataSet$gravidade)
div<-sample.split(DataSet$gravidade, SplitRatio =  0.80)
train.set<-subset(DataSet, div == TRUE)
test.set<-subset(DataSet, div == FALSE)
tamanho.test.set = length(test.set$gravidade)
tamanho.DataSet = length(DataSet$gravidade)

neuralModel <- neuralnet(gravidade~form+densidade, train.set,linear.output = TRUE, hidden = c(5,2), rep = 3, threshold = 0.001)
plot(neuralModel)

teste1 <- data.frame(test.set$form, test.set$densidade)

results <- compute(neuralModel, teste1)
results <- round(results$net.result)
Matriz_Mamografia_K_Fold_MLP <- table(results,test.set$gravidade)
print(Matriz_Mamografia_K_Fold_MLP)

# Rede Neural MLP arquitetura 01
metricas = matrix(data = 0, nrow =4 , ncol = 5)
subset_size = round(tamanho.DataSet/5)
k = 5
for (i in 1:k) {
  
  
  initLinha = 1 + (i - 1) * subset_size 
  fimLinha  = subset_size * i
  # subSetTest = DataSet[initLinha:fimLinha,]
  separacao.set <- rep(FALSE, tamanho.DataSet )
  separacao.set[initLinha:fimLinha] <- TRUE
  test.set <- DataSet[separacao.set,]
  train.set <- DataSet[!separacao.set,]
  
  
  model_MLP1 <- neuralnet(gravidade~form+densidade, train.set, linear.output = TRUE, hidden = c(9,5), rep = 3, threshold = 0.001)
  teste1_MLP1<- data.frame(test.set$form, test.set$densidade)
  results_MLP1 = compute(model_MLP1,teste1_MLP1)
  results_MLP1 <- round(results_MLP1$net.result)
  Matriz_Mamografia_K_Fold_MLP <- table(results_MLP1,test.set$gravidade)
  Precisao_Mamografia_K_Fold_MLP =  Matriz_Mamografia_K_Fold_MLP[2,2] / (Matriz_Mamografia_K_Fold_MLP[2,2] + Matriz_Mamografia_K_Fold_MLP[1,2])
  Acurracia_Mamografia_K_Fold_MLB =  (Matriz_Mamografia_K_Fold_MLP[1,1] + Matriz_Mamografia_K_Fold_MLP[2,2]) / (Matriz_Mamografia_K_Fold_MLP[1,1] + Matriz_Mamografia_K_Fold_MLP[1,2] + Matriz_Mamografia_K_Fold_MLP[2,1] + Matriz_Mamografia_K_Fold_MLP[2,2])
  Recall_Mamografia_K_Fold_MLB =  Matriz_Mamografia_K_Fold_MLP[2,2] / (Matriz_Mamografia_K_Fold_MLP[2,2] + Matriz_Mamografia_K_Fold_MLP[2,1])
  Especificidade_Mamografia_K_Fold_MLB =  Matriz_Mamografia_K_Fold_MLP[1,1] / (Matriz_Mamografia_K_Fold_MLP[1,1] + Matriz_Mamografia_K_Fold_MLP[1,2])
  metricas[1,i] =  Precisao_Mamografia_K_Fold_MLP
  metricas[2,i] =  Acurracia_Mamografia_K_Fold_MLB
  metricas[3,i] =  Recall_Mamografia_K_Fold_MLB
  metricas[4,i] =  Especificidade_Mamografia_K_Fold_MLB
}
# Rede Neural MLP arquitetura 02
metricas2 = matrix(data = 0, nrow =4 , ncol = 5)
subset_size2 = round(tamanho.DataSet/5)
k = 5
for (i in 1:k) {
  
  
  initLinha = 1 + (i - 1) * subset_size2 
  fimLinha  = subset_size2 * i
  # subSetTest = DataSet[initLinha:fimLinha,]
  separacao.set <- rep(FALSE, tamanho.DataSet )
  separacao.set[initLinha:fimLinha] <- TRUE
  test.set <- DataSet[separacao.set,]
  train.set <- DataSet[!separacao.set,]
  
  
  model_MLP2 <- neuralnet(gravidade~form+densidade, train.set, linear.output = TRUE, hidden = c(9,5,3,2), rep = 3, threshold = 0.001)
  teste1_MLP2<- data.frame(test.set$form, test.set$densidade)
  results_MLP2 = compute(model_MLP2,teste1_MLP1)
  results_MLP2 <- round(results_MLP2$net.result)
  Matriz_Mamografia_K_Fold_MLP2 <- table(results_MLP2,test.set$gravidade)
  Precisao_Mamografia_K_Fold_MLP2 =  Matriz_Mamografia_K_Fold_MLP2[2,2] / (Matriz_Mamografia_K_Fold_MLP2[2,2] + Matriz_Mamografia_K_Fold_MLP2[1,2])
  Acurracia_Mamografia_K_Fold_MLP2 =  (Matriz_Mamografia_K_Fold_MLP2[1,1] + Matriz_Mamografia_K_Fold_MLP2[2,2]) / (Matriz_Mamografia_K_Fold_MLP2[1,1] + Matriz_Mamografia_K_Fold_MLP2[1,2] + Matriz_Mamografia_K_Fold_MLP2[2,1] + Matriz_Mamografia_K_Fold_MLP2[2,2])
  Recall_Mamografia_K_Fold_MLp2 =  Matriz_Mamografia_K_Fold_MLP2[2,2] / (Matriz_Mamografia_K_Fold_MLP2[2,2] + Matriz_Mamografia_K_Fold_MLP2[2,1])
  Especificidade_Mamografia_K_Fold_MLP2 =  Matriz_Mamografia_K_Fold_MLP2[1,1] / (Matriz_Mamografia_K_Fold_MLP2[1,1] + Matriz_Mamografia_K_Fold_MLP2[1,2])
  metricas2[1,i] =  Precisao_Mamografia_K_Fold_MLP2
  metricas2[2,i] =  Acurracia_Mamografia_K_Fold_MLP2
  metricas2[3,i] =  Recall_Mamografia_K_Fold_MLp2
  metricas2[4,i] =  Especificidade_Mamografia_K_Fold_MLP2
}


#Rede Neural RBF 01 
metricas_Mamografia_K_Fold_RBF = matrix(data = 0, nrow =4 , ncol = 5)
subset_size = round(tamanho.DataSet/5)
k = 5
for (i in 1:k) {
  
   
  initLinha = 1 + (i - 1) * subset_size 
  fimLinha  = subset_size * i
  separacao.set <- rep(FALSE, tamanho.DataSet )
  separacao.set[initLinha:fimLinha] <- TRUE
  test.set <- DataSet[separacao.set,]
  train.set <- DataSet[!separacao.set,]
  
  train.set.Class <- data.frame(test.set$gravidade)
  train.set= data.frame(test.set$form, test.set$densidade)
  model_RBF1 <- rbf(train.set, train.set.Class, size = c(6),linOut = TRUE)
  
  results <- predict(model_RBF1,test.set[,1:2])
  results_RBF1<- round(results)
  
  Matriz_DataSet_K_Fold_RBF <- table(results,test.set[,6])
  Matriz_DataSet_K_Fold_RBF <- table(results_MLP1,test.set$gravidade)
  Precisao_DataSet_K_Fold_RBF =  Matriz_DataSet_K_Fold_RBF[2,2] / (Matriz_DataSet_K_Fold_RBF[2,2] + Matriz_DataSet_K_Fold_RBF[1,2])
  Acurracia_DataSet_K_Fold_RBF =  (Matriz_DataSet_K_Fold_RBF[1,1] + Matriz_DataSet_K_Fold_RBF[2,2]) / (Matriz_DataSet_K_Fold_RBF[1,1] + Matriz_DataSet_K_Fold_RBF[1,2] + Matriz_DataSet_K_Fold_RBF[2,1] + Matriz_DataSet_K_Fold_RBF[2,2])
  Recall_DataSet_K_Fold_RBF =  Matriz_DataSet_K_Fold_RBF[2,2] / (Matriz_DataSet_K_Fold_RBF[2,2] + Matriz_DataSet_K_Fold_RBF[2,1])
  Especificidade_DataSet_K_Fold_RBF =  Matriz_DataSet_K_Fold_RBF[1,1] / (Matriz_DataSet_K_Fold_RBF[1,1] + Matriz_DataSet_K_Fold_RBF[1,2])
  metricas_Mamografia_K_Fold_RBF[1,i] =  Precisao_DataSet_K_Fold_RBF
  metricas_Mamografia_K_Fold_RBF[2,i] =  Acurracia_DataSet_K_Fold_RBF
  metricas_Mamografia_K_Fold_RBF[3,i] =  Recall_DataSet_K_Fold_RBF
  metricas_Mamografia_K_Fold_RBF[4,i] =  Especificidade_DataSet_K_Fold_RBF
  #f (i == 5)
  #{
  #  metricas_Mamografia_K_Fold_RBF
  #  plot(model_RBF1)
  #}
}

#Rede Neural RBF 02 
metricas_Mamografia_K_Fold_RBF2 = matrix(data = 0, nrow =4 , ncol = 5)
subset_size2 = round(tamanho.DataSet/5)
k = 5
for (i in 1:k) {
  
  
  initLinha = 1 + (i - 1) * subset_size2 
  fimLinha  = subset_size2 * i
  separacao.set <- rep(FALSE, tamanho.DataSet )
  separacao.set[initLinha:fimLinha] <- TRUE
  test.set <- DataSet[separacao.set,]
  train.set <- DataSet[!separacao.set,]
  
  train.set.Class <- data.frame(test.set$gravidade)
  train.set= data.frame(test.set$form, test.set$densidade)
  model_RBF2 <- rbf(train.set, train.set.Class, size = c(9),linOut = TRUE)
  
  results <- predict(model_RBF2,test.set[,1:2])
  results_RBF2<- round(results)
  
  Matriz_DataSet_K_Fold_RBF2 <- table(results,test.set[,6])
  Matriz_DataSet_K_Fold_RBF2 <- table(results_MLP2,test.set$gravidade)
  Precisao_DataSet_K_Fold_RBF2 =  Matriz_DataSet_K_Fold_RBF2[2,2] / (Matriz_DataSet_K_Fold_RBF2[2,2] + Matriz_DataSet_K_Fold_RBF2[1,2])
  Acurracia_DataSet_K_Fold_RBF2 =  (Matriz_DataSet_K_Fold_RBF2[1,1] + Matriz_DataSet_K_Fold_RBF2[2,2]) / (Matriz_DataSet_K_Fold_RBF2[1,1] + Matriz_DataSet_K_Fold_RBF2[1,2] + Matriz_DataSet_K_Fold_RBF2[2,1] + Matriz_DataSet_K_Fold_RBF2[2,2])
  Recall_DataSet_K_Fold_RBF2 =  Matriz_DataSet_K_Fold_RBF2[2,2] / (Matriz_DataSet_K_Fold_RBF2[2,2] + Matriz_DataSet_K_Fold_RBF2[2,1])
  Especificidade_DataSet_K_Fold_RBF2 =  Matriz_DataSet_K_Fold_RBF2[1,1] / (Matriz_DataSet_K_Fold_RBF2[1,1] + Matriz_DataSet_K_Fold_RBF2[1,2])
  metricas_Mamografia_K_Fold_RBF2[1,i] =  Precisao_DataSet_K_Fold_RBF2
  metricas_Mamografia_K_Fold_RBF2[2,i] =  Acurracia_DataSet_K_Fold_RBF2
  metricas_Mamografia_K_Fold_RBF2[3,i] =  Recall_DataSet_K_Fold_RBF2
  metricas_Mamografia_K_Fold_RBF2[4,i] =  Especificidade_DataSet_K_Fold_RBF2
  #f (i == 5)
  #{
  #  metricas_Mamografia_K_Fold_RBF
  #  plot(model_RBF1)
  #}
}

#IMPRIMIR 
print("++++++++++++++++ARQUITETURA MLP++++++++++++")
metricas
metricas2
print("++++++++++++++++ARQUITETURA RBF++++++++++++")
metricas_Mamografia_K_Fold_RBF
metricas_Mamografia_K_Fold_RBF2
