#load("clase_svm/df17.RData")
#load("clase_svm/df56.RData")

#Cargar datos de entrenamiento
trainingData <- read.csv("./data/mnist_train.csv",header = FALSE)

#Cargar datos de prueba
testingData <- read.csv("./data/mnist_test.csv",header = FALSE)


#Ver uno de los digitos como imagen
library(magick)
im <- function(x,m){ #Funcion para ver el digito
  plot(image_read(aperm(array(as.numeric(m[x,2:785]),c(28,28,1)),c(2,1,3)))) #Muestra la imagen en el viewer
  return(m[x,1]) #Muestra el numero respectivo en consola
}
#Imagen 4 de trainingData
im(10,trainingData)


#Imagen 8 de testingData
im(8,testingData)



#********************Descriptiva**************************#

# dimension de las bases de datos.
dim(trainingData)
dim(testingData)

# distribucion de los digitos en la base de datos de entrenamiento 
#pdf(file = "barras_digitos_entrenamiento.pdf", height = 4.5, width = 6.5)
barplot(table(trainingData$V1))
#dev.off()


# distribucion de los digitos en la base de datos de validacion
#pdf(file="barras_digitos_validacion.pdf",  height = 4.5, width = 6.5)
barplot(table(testingData$V1))
#dev.off()

#****************** Preparando datos train (muestra) ***************#
# transformemos las vriables a numericas
trainingData <- apply(trainingData, 2, as.numeric)
trainingData <- as.data.frame(trainingData)
trainingData$V1 <- as.factor(trainingData$V1)
#apply(trainingData, 2, sum)


# tomemos una muestra por que mi pc no aguanta
muestra <- sample(1:6000, 5000)
trainingData2 <- as.data.frame(trainingData[muestra,])



#****************** Preparando datos test (muestra) ***************#
# transformemos las vriables a numericas
testingData <- apply(testingData, 2, as.numeric)
testingData <- as.data.frame(testingData)
testingData$V1 <- as.factor(testingData$V1)






# *************** Modelos ***********************#
# 1) regresion logistica multinomial -----------------------------------------

library(nnet)
# máximo número de iteraciones igual a 100 (por defecto)
lm_1 <- multinom(V1 ~ .,data = trainingData2, MaxNWts = 7860)

# máximo número de iteraciones igual a 200 (maxit=300)
lm2 <- multinom(V1 ~ .,data = trainingData2, MaxNWts = 7860, maxit = 200)

# máximo número de iteraciones igual a 300 (maxit=300) y tolerancia de 1.e-8
lm3 <- multinom(V1 ~ .,data = trainingData2, MaxNWts = 7860, maxit = 200, abstol = 1.e-8)


pred_valid <- predict(lm_1, newdata = testingData)
matrizConfusion_class <- table(testingData$V1, pred_valid)
matrizConfusion_class
clasificacion_correctas_class <- sum(diag(matrizConfusion_class)) / 10000

library(xtable)
xtable(matrizConfusion_class)


#2) Arboles de clasificacion ------------------------------------------------

library(rpart)

# Arboles de clasifiación con los metodos class
modelo1_rpart <- rpart(V1 ~ ., data = trainingData, method = "class")

# predict sobre los datos de prueba.
pred_valid <- predict(modelo3_rpart, newdata = testingData)

# matriz de confusión
matrizConfusion_class <- table(testingData$V1, pred_valid)
matrizConfusion_class
clasificacion_correctas_class <- sum(diag(matrizConfusion_class)) / 10000



# 3) Bosques aleatorio ----------------------------------------------------------

# construccion de bosques aleatorio con 10, 100 y 200 árboles
library(randomForest)
clasificadorRF1 <- randomForest(V1 ~ ., data = trainingData, ntree = 10)
clasificadorRF2 <- randomForest(V1 ~ ., data = trainingData, ntree = 100)
clasificadorRF3 <- randomForest(V1 ~ ., data = trainingData, ntree = 200)

# calculando matriz de confusion y tasa clasificacion correcta con datos de prueba 
# aqui se calcula con el modelo clasificadorRF1, analogamente se hace con los demas modelos
pred_valid_RF <- predict(clasificadorRF1, newdata = testingData)
matrizConfusion <- table(testingData$V1, pred_valid_RF)
matrizConfusion
clasificacion_correctas <- sum(diag(matrizConfusion)) / 10000
clasificacion_correctas




# 4) Máquinas de soporte vectorial ----------------------------------------------

# cargando la librería al ambiente de trabajo
library(e1071)
# hacemos uso de la función svm con kernel lineal y tipo clasificación
clasificadorSVM_lineal <- svm(V1 ~ ., data = trainingData, 
                       type = 'C-classification', kernel = 'linear')

# hacemos uso de la función svm con kernel polynomial y tipo clasificación
clasificadorSVM_polynomial <- svm(V1 ~ ., data = trainingData, 
                              type = 'C-classification', kernel = 'polynomial')


# haciendo predicción sobre el conjunto de validación
pred_valid <- predict(clasificadorSVM_lineal, newdata = testingData)

# calculando la matriz de confusión
matrizConfusion <- table(testingData$V1, pred_valid)

# calculando la tasa de clasificación de observaciones correctas
clasificacion_correctas <- sum(diag(matrizConfusion)) / 10000







# 5) Redes neuronales ----------------------------------------------------

load("model1.rdata")



library(keras)

#Cargar datos de entrenamiento
trainingData <- read.csv("./data/mnist_train.csv",header = FALSE)

#Cargar datos de prueba
testingData <- read.csv("./data/mnist_test.csv",header = FALSE)

# Labels (columna 1)
trainY <- trainingData["V1"] %>% as.matrix() %>% as.numeric()
trainLabels <- to_categorical(trainY)
testY <- testingData["V1"] %>% as.matrix() %>% as.numeric()
testLabels <- to_categorical(testY)

# Transformar los datos, para el modelo 1 (acepta vectores)
trainingData <- trainingData %>%
  subset(select = -c(V1)) %>%
  as.matrix() %>%
  as.numeric() %>%
  array(c(60000,784))

testingData <- testingData %>%
  subset(select = -c(V1)) %>%
  as.matrix() %>%
  as.numeric() %>%
  array(c(10000,784))

# Transformar los datos para el modelo 2 (acepta matrices)
trainingData2D <- array(trainingData, c(60000,28,28,1))

testingData2D <- array(testingData, c(10000,28,28,1))

#Ver uno de los digitos como imagen
library(magick)
im <- function(x,m,n){ #Funcion para ver el digito
  plot(image_read(aperm(array(m[x,],c(28,28,1)),c(2,1,3)))) #Muestra la imagen en el viewer
  return(n[x]) #Muestra el numero respectivo en consola
}
#Imagen 4 de trainingData
im(4,trainingData,trainY)
#Imagen 8 de testingData
im(8,testingData,testY)

# Primer modelo, tomado de: https://nextjournal.com/gkoehler/digit-recognition-with-keras
model1 <- keras_model_sequential() %>%
  layer_dense(512, input_shape=c (784,NULL), activation = 'relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(128, activation = 'relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(10, activation = 'softmax') %>%
  compile(loss='categorical_crossentropy', metrics='accuracy', optimizer='adam')

# Entrenar primer modelo
history1 <- model1 %>%
  fit(trainingData,
      trainLabels,
      epochs = 100,
      batch_size = 2048,
      validation_split = 0.2)
#validation_data = list(test, testLabels))
plot(history)

# Segundo modelo, tomado de: https://medium.com/coinmonks/handwritten-digit-prediction-using-convolutional-neural-networks-in-tensorflow-with-keras-and-live-5ebddf46dc8

model2 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(5,5), input_shape = c(28,28,1), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size=c(2, 2)) %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size=c(2, 2)) %>%
  layer_dropout(0.2) %>%
  layer_flatten() %>%
  layer_dense(128, activation='relu') %>%
  layer_dense(10, activation='softmax') %>%
  compile(loss='categorical_crossentropy', optimizer='adam', metrics=c('accuracy'))

# Entrenar segundo modelo
history2 <- model2 %>% 
  fit(trainingData2D,
      trainLabels,
      validation_split = 0.2,
      epochs=10,
      batch_size=200)


#Guardar
save_model_hdf5(model1, "model1")
save_model_hdf5(model2, "model2")

#Cargar
model1 <- load_model_hdf5("model1")
model2 <- load_model_hdf5("model2")

#############################EVALUACION MODELO######################################
evalModel <- function(model, dataSet, labelSet, Y){
  # Evaluation & Prediction - train data
  print(model %>% evaluate(dataSet, labelSet))
  predtrain <- model %>% predict_classes(dataSet)
  a <- table(Predicted = predtrain, Actual = Y)
  print(a)
  prob <- model %>% predict_proba(dataSet)
  # print(cbind(prob, Predicted_class = predtrain, Actual = Y))
  
  # Conjunto de imagenes de entrenamiento mal predichas
  badPredictions <- cbind(predicted = predtrain, correct = Y, pos = c(1:length(Y)))
  badPredictions <- subset(badPredictions, badPredictions[,1] != badPredictions[,2])
  return(badPredictions)
}

# Funcion para ver uno de los digitos mal predichos
library(magick)
inspectBadPred <- function(predictions,dataset,pos){ #Funcion para ver el digito
  plot(image_read(aperm(array(dataset[predictions[pos,3],],c(28,28,1)),c(2,1,3)))) #Muestra la imagen en el viewer
  print(paste("predicho:",predictions[pos,1]))
  print(paste("real:    ",predictions[pos,2]))
}

# Modelo 1 con datos de entrenamiento
badPredTrain1 <- evalModel(model1,trainingData,trainLabels,trainY)

# Modelo 2 con datos de entrenamiento
badPredTrain2 <- evalModel(model2,trainingData2D,trainLabels,trainY)

# Modelo 1 con datos de prueba
badPredTest1 <- evalModel(model1,testingData,testLabels,testY)

# Modelo 2 con datos de prueba
badPredTest2 <- evalModel(model2,testingData2D,testLabels,testY)

# Ver primera prediccion errada del modelo 2 con datos de prueba
inspectBadPred(badPredTest2,testingData,1)

# Ver segunda prediccion errada del modelo 1 con datos de entrenamiento
inspectBadPred(badPredTrain1,trainingData,2)



load_model_hdf5("model1")

load("confusion.RData")

tasa <- sum(diag(confusionTest2)) / sum(confusionTest2)

library(xtable)
xtable(confusionTest2)





