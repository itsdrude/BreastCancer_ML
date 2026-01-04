import pandas as pd
import numpy as np
import tensorflow.keras as kr
import matplotlib.pyplot as plt
import time

#===============================================================================
#                                   BLOCCO 1
#===============================================================================
start_execution = time.time()

# Lettura del file csv
df = pd.read_csv('data/Breast_Cancer.csv')

# Formato dei dati di input
dataset = df.values # converte il csv in una matrice array

in_a = dataset[:, 0] # età
in_b = dataset[:, 1] # razza
in_c = dataset[:, 2] # stato civile
in_d = dataset[:, 3] # stadio T
in_e = dataset[:, 4] # stadio N
in_f = dataset[:, 5] # 6º stadio
in_g = dataset[:, 6] # differenziazione
in_h = dataset[:, 7] # grado
in_i = dataset[:, 8] # stadio A
in_j = dataset[:, 9] # dimensione del tumore
in_k = dataset[:, 10] # stato degli estrogeni
in_l = dataset[:, 11] # stato dei progestageni
in_m = dataset[:, 12] # esame del nodo regionale
in_n = dataset[:, 13] # nodo retinolo positivo
in_o = dataset[:, 14] # mesi di sopravvivenza
in_p = dataset[:, 15] # stato vitale vivo/morto

for i in range(len(in_b)): #White = 0, Black = 1, Other = 2
  if in_b[i] == 'White':
    in_b[i] = 0
  if in_b[i] == 'Black':
    in_b[i] = 1
  else:
    in_b[i] = 2

for i in range(len(in_c)): #Married = 0, Divorced = 1, Single = 2, Widowed = 3, Separated = 4
  if in_c[i] == 'Married':
    in_c[i] = 0
  if in_c[i] == 'Divorced':
    in_c[i] = 1
  if in_c[i] == 'Single':
    in_c[i] = 2
  if in_c[i] == 'Single ': # Teniamo conto di alcuni errori nel dataset
    in_c[i] = 2
  if in_c[i] == 'Widowed':
    in_c[i] = 3
  else:
    in_c[i] = 4

for i in range(len(in_d)): #T1 = 0, T2 = 1, T3 = 2, T4 = 3
  if in_d[i] == 'T1':
    in_d[i] = 0
  if in_d[i] == 'T2':
    in_d[i] = 1
  if in_d[i] == 'T3':
    in_d[i] = 2
  else:
    in_d[i] = 3

for i in range(len(in_e)): #N1 = 0, N2 = 1, N3 = 2
  if in_e[i] == 'N1':
    in_e[i] = 0
  if in_e[i] == 'N2':
    in_e[i] = 1
  else:
    in_e[i] = 2

for i in range(len(in_f)): #IIA = 0, IIB = 1, IIIA = 2, IIIB = 3, IIIC = 4
  if in_f[i] == 'IIA':
    in_f[i] = 0
  if in_f[i] == 'IIB':
    in_f[i] = 1
  if in_f[i] == 'IIIA':
    in_f[i] = 2
  if in_f[i] == 'IIIB':
    in_f[i] = 3
  else:
    in_f[i] = 4

for i in range(len(in_g)): #Poorly differenciated = 0, Moderately differenciated = 1, Well differenciated = 2, Undifferenciated = 3
  if in_g[i] == 'Poorly differentiated':
    in_g[i] = 0
  if in_g[i] == 'Moderately differentiated':
    in_g[i] = 1
  if in_g[i] == 'Well differentiated':
    in_g[i] = 2
  else:
    in_g[i] = 3

for i in range(len(in_h)): #1 = 0, 2 = 1, 3 = 2, anaplastic; Grade IV = 3
  if in_h[i] == '1':
    in_h[i] = 0
  if in_h[i] == '2':
    in_h[i] = 1
  if in_h[i] == '3':
    in_h[i] = 2
  else:
    in_h[i] = 3

for i in range(len(in_i)): #Regional = 0, Distant = 1
  if in_i[i] == 'Regional':
    in_i[i] = 0
  else:
    in_i[i] = 1

for i in range(len(in_k)): #Positive = 0, Negative = 1
  if in_k[i] == 'Positive':
    in_k[i] = 0
  else:
    in_k[i] = 1

for i in range(len(in_l)): #Positive = 0, Negative = 1
  if in_l[i] == 'Positive':
    in_l[i] = 0
  else:
    in_l[i] = 1

for i in range(len(in_p)): #Dead = 0, Alive = 1
  if in_p[i] == 'Dead':
    in_p[i] = 0
  else:
    in_p[i] = 1

# Riassembliamo tutti i dati e mescoliamo le righe della matrice.
df_new = np.transpose(np.array([in_a, in_b, in_c, in_d, in_e, in_f, in_g, in_h, in_i, in_j, in_k, in_l, in_m, in_n, in_o, in_p])) # Forma = (4024, 15)
np.random.shuffle(df_new) # mescolare i pazienti

# Prendiamo il 70% del dataset per l'allenamento e il 30% per il test
lines_train = int(df_new.shape[0]*0.70) # numero di righe per l'allenamento

data_train = df_new[:lines_train] # da 0 a lines_train (range dei dati dallo 0 al 70%)
data_test = df_new[lines_train:df_new.shape[0]] # da lines_train alla fine (range dei dati dal 70 al 100%)

# Creiamo il vettore di input e output per allenamento e test
Y_train = data_train[:, -1] # output
X_train = data_train[:, :-1] # input

Y_test = data_test[:, -1]
X_test = data_test[:, :-1]

# Convertiamo gli elementi dei due array dei due set a int per evitare
# valori float non necessari (tutti devono essere interi come definiti sopra)
X_train = X_train.astype('int')
Y_train = Y_train.astype('int')
X_test = X_test.astype('int')
Y_test = Y_test.astype('int')

#===============================================================================
#                                   BLOCCO 2
#===============================================================================

# Definiamo il modello e i suoi parametri che sarà creato con Keras
learning_rate = 0.01 # tasso di apprendimento
n_neurons = [15, 10, 10, 1] # numero di neuroni per strato, uno per parametro 15 primo strato - 16 strati intermedi - 1 ultimo strato (probabilità)

model = kr.Sequential() # Creiamo un modello che consiste in una sequenza di strati

# Creiamo i 4 strati e li aggiungiamo al modello (Dense indica che un neurone del livello precedente è collegato a tutti quelli del livello successivo)
model.add(kr.layers.Dense(n_neurons[1], activation='relu')) # primo strato nascosto
model.add(kr.layers.Dense(n_neurons[2], activation='relu'))
# model.add(kr.layers.Dense(n_neurons[3], activation='relu'))
model.add(kr.layers.Dense(n_neurons[3], activation='sigmoid')) # <-- funzione di attivazione di strato di uscita

# Compiliamo il modello (necessario per Keras)
model.compile(loss='binary_crossentropy',
              optimizer=kr.optimizers.SGD(learning_rate=learning_rate),
              metrics=['accuracy'])


history = model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test))

#===============================================================================
#                                   BLOCCO 3
#===============================================================================

plt.figure(figsize=(12, 5))

# Grafico della funzione di perdita
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Test')
plt.title('Función error del modelo')
plt.xlabel('Ciclos')
plt.ylabel('Errores')
plt.legend()
plt.ylim()

# Grafico della precisione
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Test')
plt.title('Precisión del modelo')
plt.xlabel('Ciclos')
plt.ylabel('Presición')
plt.legend()
plt.ylim(0.83, 0.95)

print('Final loss function value for test data: ', round(history.history['val_loss'][-1], 3))
print('Final loss function value for train data: ', round(history.history['loss'][-1], 3))
print('Predictive accuracy of the model: ', round(history.history['val_accuracy'][-1]*100, 3), '%')

end_execution = time.time()
print('Execution time: ', round(end_execution - start_execution, 3), ' seconds')


plt.show()