library(e1071)
library(mlbench)
library(caTools)
library(caret)
library(klaR)
library(neuralnet)


DataSet <- read.csv('Heart.csv')
colnames(DataSet)=c('Age','sex','cp','trestbps','col','fbs','restecg','thalach','exang','pico.antigo','inclinacao','ca','thal','Num')

set.seed(2)
DataSet$Num =  as.numeric(DataSet$Num)
div<-sample.split(DataSet$Num, SplitRatio =  0.80)
train.set<-subset(DataSet, div == TRUE)
test.set<-subset(DataSet, div == FALSE)
tamanho.test.set = length(test.set$Num)
tamanho.DataSet = length(DataSet$Num)

neuralModel <- neuralnet(Num~sex+cp, train.set,linear.output = TRUE, hidden = c(5,2), rep = 3, threshold = 0.001)
plot(neuralModel)

teste1 <- data.frame(test.set$sex, test.set$cp)

View(teste1)


results <- compute(neuralModel, teste1)
results <- round(results$net.result)
Matriz_CMC_K_Fold_MLP <- table(results,test.set$Num)
print(Matriz_CMC_K_Fold_MLP)

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
  
  
  model_MLP1 <- neuralnet(Num~sex+cp, train.set, linear.output = TRUE, hidden = c(9,5), rep = 3, threshold = 0.001)
  teste1_MLP1<- data.frame(test.set$sex, test.set$cp)
  results_MLP1 = compute(model_MLP1,teste1_MLP1)
  results_MLP1 <- round(results_MLP1$net.result)
  Matriz_CMC_K_Fold_MLP <- table(results_MLP1,test.set$Num)
  Precisao_CMC_K_Fold_MLP =  Matriz_CMC_K_Fold_MLP[2,2] / (Matriz_CMC_K_Fold_MLP[2,2] + Matriz_CMC_K_Fold_MLP[1,2])
  Acurracia_CMC_K_Fold_MLB =  (Matriz_CMC_K_Fold_MLP[1,1] + Matriz_CMC_K_Fold_MLP[2,2]) / (Matriz_CMC_K_Fold_MLP[1,1] + Matriz_CMC_K_Fold_MLP[1,2] + Matriz_CMC_K_Fold_MLP[2,1] + Matriz_CMC_K_Fold_MLP[2,2])
  Recall_CMC_K_Fold_MLB =  Matriz_CMC_K_Fold_MLP[2,2] / (Matriz_CMC_K_Fold_MLP[2,2] + Matriz_CMC_K_Fold_MLP[2,1])
  Especificidade_CMC_K_Fold_MLB =  Matriz_CMC_K_Fold_MLP[1,1] / (Matriz_CMC_K_Fold_MLP[1,1] + Matriz_CMC_K_Fold_MLP[1,2])
  metricas[1,i] =  Precisao_CMC_K_Fold_MLP
  metricas[2,i] =  Acurracia_CMC_K_Fold_MLB
  metricas[3,i] =  Recall_CMC_K_Fold_MLB
  metricas[4,i] =  Especificidade_CMC_K_Fold_MLB
}
# Rede Neural MLP arquitetura 02
metricas2 = matrix(data = 0, nrow =4 , ncol = 5)
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
  
  model_MLP2 <- neuralnet(Num~sex+cp, train.set, linear.output = TRUE, hidden = c(9,5,2), rep = 3, threshold = 0.001)
  teste1_MLP2<- data.frame(test.set$sex, test.set$cp)
  results_MLP2 = compute(model_MLP2,teste1_MLP2)
  results_MLP2 <- round(results_MLP2$net.result)
  Matriz_CMC_K_Fold_MLP2 <- table(results_MLP2,test.set$Num)
  Precisao_CMC_K_Fold_MLP2 =  Matriz_CMC_K_Fold_MLP2[2,2] / (Matriz_CMC_K_Fold_MLP2[2,2] + Matriz_CMC_K_Fold_MLP2[1,2])
  Acurracia_CMC_K_Fold_MLP2 =  (Matriz_CMC_K_Fold_MLP2[1,1] + Matriz_CMC_K_Fold_MLP2[2,2]) / (Matriz_CMC_K_Fold_MLP2[1,1] + Matriz_CMC_K_Fold_MLP2[1,2] + Matriz_CMC_K_Fold_MLP2[2,1] + Matriz_CMC_K_Fold_MLP2[2,2])
  Recall_CMC_K_Fold_MLP2 =  Matriz_CMC_K_Fold_MLP2[2,2] / (Matriz_CMC_K_Fold_MLP2[2,2] + Matriz_CMC_K_Fold_MLP2[2,1])
  Especificidade_CMC_K_Fold_MLP2 =  Matriz_CMC_K_Fold_MLP2[1,1] / (Matriz_CMC_K_Fold_MLP2[1,1] + Matriz_CMC_K_Fold_MLP2[1,2])
  metricas2[1,i] =  Precisao_CMC_K_Fold_MLP2
  metricas2[2,i] =  Acurracia_CMC_K_Fold_MLP2
  metricas2[3,i] =  Recall_CMC_K_Fold_MLP2
  metricas2[4,i] =  Especificidade_CMC_K_Fold_MLP2
}

#Rede Neural RBF arquitetura 01
metricas_DataSet_K_Fold_RBF = matrix(data = 0, nrow =4 , ncol = 5)
subset_size = round(tamanho.DataSet/5)
k = 5
for (i in 1:k) {
  
 
  initLinha = 1 + (i - 1) * subset_size 
  fimLinha  = subset_size * i
  separacao.set <- rep(FALSE, tamanho.DataSet)
  separacao.set[initLinha:fimLinha] <- TRUE
  test.set <- DataSet[separacao.set,]
  train.set <- DataSet[!separacao.set,]
  
  train.set.Num <- data.frame(train.set$Num)
  trainParametro.set= data.frame(train.set$sex, train.set$cp)
  testParametro.set= data.frame(test.set$sex, test.set$cp)
  
  model_RBF1 <- rbf(trainParametro.set, train.set.Num, size = c(6),linOut = TRUE)
  
  results <- predict(model_RBF1,testParametro.set)
  results_RBF1<- round(results)
  
  Matriz_DataSet_K_Fold_RBF <- table(results_RBF1,test.set$Num)
  Precisao_DataSet_K_Fold_RBF =  Matriz_DataSet_K_Fold_RBF[2,2] / (Matriz_DataSet_K_Fold_RBF[2,2] + Matriz_DataSet_K_Fold_RBF[1,2])
  Acurracia_DataSet_K_Fold_RBF =  (Matriz_DataSet_K_Fold_RBF[1,1] + Matriz_DataSet_K_Fold_RBF[2,2]) / (Matriz_DataSet_K_Fold_RBF[1,1] + Matriz_DataSet_K_Fold_RBF[1,2] + Matriz_DataSet_K_Fold_RBF[2,1] + Matriz_DataSet_K_Fold_RBF[2,2])
  Recall_DataSet_K_Fold_RBF =  Matriz_DataSet_K_Fold_RBF[2,2] / (Matriz_DataSet_K_Fold_RBF[2,2] + Matriz_DataSet_K_Fold_RBF[2,1])
  Especificidade_DataSet_K_Fold_RBF =  Matriz_DataSet_K_Fold_RBF[1,1] / (Matriz_DataSet_K_Fold_RBF[1,1] + Matriz_DataSet_K_Fold_RBF[1,2])
  metricas_DataSet_K_Fold_RBF[1,i] =  Precisao_DataSet_K_Fold_RBF
  metricas_DataSet_K_Fold_RBF[2,i] =  Acurracia_DataSet_K_Fold_RBF
  metricas_DataSet_K_Fold_RBF[3,i] =  Recall_DataSet_K_Fold_RBF
  metricas_DataSet_K_Fold_RBF[4,i] =  Especificidade_DataSet_K_Fold_RBF
 
}

#Rede Neural RBF arquitetura 02
metricas_DataSet_K_Fold_RBF2 = matrix(data = 0, nrow =4 , ncol = 5)
subset_size2 = round(tamanho.DataSet/5)
k = 5
for (i in 1:k) {
  
  
  initLinha = 1 + (i - 1) * subset_size2 
  fimLinha  = subset_size2 * i
  separacao.set <- rep(FALSE, tamanho.DataSet)
  separacao.set[initLinha:fimLinha] <- TRUE
  test.set <- DataSet[separacao.set,]
  train.set <- DataSet[!separacao.set,]
  
  train.set.Num <- data.frame(train.set$Num)
  trainParametro.set= data.frame(train.set$sex, train.set$cp)
  testParametro.set= data.frame(test.set$sex, test.set$cp)

  model_RBF2 <- rbf(trainParametro.set, train.set.Num, size = c(6,5),linOut = TRUE)
 
  results <- predict(model_RBF2,testParametro.set)
  results_RBF2<- round(results)
  
  Matriz_DataSet_K_Fold_RBF2 <- table(results_RBF2,test.set$Num)
  Precisao_DataSet_K_Fold_RBF2 =  Matriz_DataSet_K_Fold_RBF2[2,2] / (Matriz_DataSet_K_Fold_RBF2[2,2] + Matriz_DataSet_K_Fold_RBF2[1,2])
  Acurracia_DataSet_K_Fold_RBF2 =  (Matriz_DataSet_K_Fold_RBF2[1,1] + Matriz_DataSet_K_Fold_RBF2[2,2]) / (Matriz_DataSet_K_Fold_RBF2[1,1] + Matriz_DataSet_K_Fold_RBF2[1,2] + Matriz_DataSet_K_Fold_RBF2[2,1] + Matriz_DataSet_K_Fold_RBF2[2,2])
  Recall_DataSet_K_Fold_RBF2 =  Matriz_DataSet_K_Fold_RBF2[2,2] / (Matriz_DataSet_K_Fold_RBF2[2,2] + Matriz_DataSet_K_Fold_RBF2[2,1])
  Especificidade_DataSet_K_Fold_RBF2 =  Matriz_DataSet_K_Fold_RBF[1,1] / (Matriz_DataSet_K_Fold_RBF[1,1] + Matriz_DataSet_K_Fold_RBF2[1,2])
  metricas_DataSet_K_Fold_RBF2[1,i] =  Precisao_DataSet_K_Fold_RBF
  metricas_DataSet_K_Fold_RBF2[2,i] =  Acurracia_DataSet_K_Fold_RBF
  metricas_DataSet_K_Fold_RBF2[3,i] =  Recall_DataSet_K_Fold_RBF
  metricas_DataSet_K_Fold_RBF2[4,i] =  Especificidade_DataSet_K_Fold_RBF
  
}



