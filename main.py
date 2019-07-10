import pandas as pd
from __future__ import print_function

import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop,SGD
from tensorflow.keras.regularizers import l1


dicionario = pd.read_csv('imdb_dicionario.csv')
corpus_text = '\n'.join(dicionario['review'])
sentences = corpus_text.split('\n')
sentences = [line.lower().split(' ') for line in sentences]



def clean(s):
    return [w.strip(',."!?:;()\';') for w in s]
sentences = [clean(s) for s in sentences if len(s) > 0]


from gensim.models import Word2Vec


model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)

vectors = model.wv
del model

#####AQUI VOU LER E ALIMENTAR OS DADOS DE TREINAMENTO#####

## lendo arquivo com comentarios que serão usados no treinamento

treinamento = pd.read_csv('imdb_treinamento.csv')


vetorIdt = []
x_treinamento = []  ##vetor vai receber as palavras e frases em forma de vetores

for i in range (0,20000):
  vetorIdt.clear()
  frase = treinamento['review'].loc[i]
  vetorPalavras = (frase.lower().split(" "))
  for j in range (0, len(vetorPalavras)):
    vetorPalavras[j] = vetorPalavras[j].strip(',."!?:;()\';')
  for k in range (0,len(vetorPalavras)):
    z=vectors[vetorPalavras[k]]
    vetorIdt.append(z)
  x_treinamento.append(vetorIdt)


#####AQUI VOU LER E ALIMENTAR OS DADOS DE TESTE#####  
  
## lendo arquivo com comentarios que serão usados no teste

teste = pd.read_csv('imdb_teste.csv')


##aqui começa a petecar

vetorId = []
x_teste = []  ##vetor vai receber as palavras e frases em forma de vetores

for i in range (0,5000):
  vetorId.clear()
  frase = teste['review'].loc[i]
  vetorPalavras = (frase.lower().split(" "))
  for j in range (0, len(vetorPalavras)):
    vetorPalavras[j] = vetorPalavras[j].strip(',."!?:;()\';')
  for k in range (0,len(vetorPalavras)):
    z=vectors[vetorPalavras[k]]
    vetorId.append(z)
  x_teste.append(vetorId)
  
print(treinamento['sentiment'].loc[0])
  

######Y TESTE#######  

y_teste = []

for i in range (0,5000):
  sent = teste['sentiment'].loc[i]
  if sent == "Positive":  
    y_teste.append(1)
  else:
    y_teste.append(0)
  
######Y TREINAMENTO#######  

y_treinamento = []

for i in range (0,20000):
  sent = treinamento['sentiment'].loc[i]
  if sent == "Positive":  
    y_treinamento.append(1)
  else:
    y_treinamento.append(0)
   
print (len(y_teste))

print (y_teste)

zero = []

for i in range (0,100):
  zero.append(0)
  
for j in range (0,20000):
  if len(x_treinamento[j]) < 470:
    x_treinamento[j].append(zero)
    

for j in range (0,5000):
  if len(x_teste[j]) < 470:
    x_teste[j].append(zero)
    
# Transformando as matrizes em vetores unidimensionais
x_treinamento = np.array(x_treinamento).reshape(20000, 47000)
x_teste = np.array(x_teste).reshape(5000, 47000)
x_treinamento = x_treinamento.astype('float32')
x_teste = x_teste.astype('float32')

    ###AQUI COMEÇO IMPLEMENTAR REDE NEURAL####
  
batch_size = 100
num_classes = 10
epochs = 500

# Definição da arquitetura do modelo
model = Sequential()
# adicione aqui as camadas do modelo
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(150, input_dim=47000,activation='relu'))
model.add(tf.keras.layers.Dense(20, activation='tanh'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))


# Fim - Definição da arquitetura do modelo

model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=SGD(0.01),
              metrics=['accuracy'])

# Treinamento do modelo 
H = model.fit(x_treinamento, y_treinamento,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_teste, y_teste))

# Avaliação do modelo no conjunto de teste
score = model.evaluate(x_teste, y_teste, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# plotando 'loss' e 'accuracy' para os datasets 'train' e 'test'
plt.figure()
#plt.plot(np.arange(0,epochs), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0,epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,epochs), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,epochs), H.history["val_acc"], label="val_acc")
plt.title("Acurácia")
plt.xlabel("Épocas #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()