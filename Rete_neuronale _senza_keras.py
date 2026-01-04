import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time as time
from IPython.display import clear_output


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



# Riassembliamo tutte le colonne formando la matrice iniziale con i nuovi valori assegnati a ciascuna categoria
df_new = np.transpose(np.array([in_a, in_b, in_c, in_d, in_e, in_f, in_g, in_h, in_i, in_j, in_k, in_l, in_m, in_n, in_o, in_p])) # Shape = (4024, 15)


# class of network layer

def stand_matrix(array):
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    return (array - mean)/std

def precision(Y_predict, Y_true):
  k = 0
  for i in range(len(Y_predict)):
    if round(Y_predict[i][0]) == Y_true[i][0]:
      k = k + 1
  return (k/len(Y_predict))*100


class capa():
  def __init__(self, inputs, neurons, activ_func, d_activ_func):
    self.a_function = activ_func # Funzione di attivazione
    self.d_a_function = d_activ_func # Derivata della funzione di attivazione

    self.b = np.random.normal(0, 1, (1, neurons)) # Bias
    self.W = np.random.normal(0, 1, (inputs, neurons)) # Pesi

def build_network(structure, activ_func, d_activ_func, activ_func_output, d_activ_func_output):
  network = []
  for i in range(len(structure)-2):
    network.append(capa(structure[i], structure[i + 1], activ_func, d_activ_func))
  network.append(capa(structure[-2], structure[-1], activ_func_output, d_activ_func_output))
  return network

def train_network(network, X, Y, cost_func, d_cost_func, learning_rate = 0.5, mode_train = True):
  output_layers = [(None, X)]

  # Propagazione in avanti (predizione)
  for i in range(len(network)):

    z = np.dot(output_layers[-1][1], network[i].W) + network[i].b # Somma ponderata dello strato
    a = network[i].a_function(z) # Uscita dello strato
    output_layers.append((z, a))

  # Propagazione all'indietro
  if mode_train:

    delta_error = []
    for i in reversed(range(0, len(network))):
      z = output_layers[i+1][0] # Somma ponderata dello strato i
      a = output_layers[i+1][1] # Uscita dello strato i

      if i == len(network)-1: # Delta dell'ultimo strato
        delta_error.insert(0, d_cost_func(a, Y) * network[i].d_a_function(z))

      else: # Delta degli strati nascosti
        delta_error.insert(0, np.dot(delta_error[0], W_.T) * network[i].d_a_function(z))

      W_ = network[i].W
      # Discesa del gradiente
      grad_W = np.mean(delta_error[0], axis=0, keepdims=True)
      grad_b = np.dot(output_layers[i][1].T, delta_error[0])
      network[i].b = network[i].b - learning_rate * grad_W
      network[i].W = network[i].W - learning_rate * grad_b

  return output_layers[-1][1]

# funzioni di attivazione

def sigmoid(x):
  return 1 / (1 + np.e ** (-x))

def d_sigmoid(x):
  return (1 / (1 + np.e ** (-x)) * (1 - (1 / (1 + np.e ** (-x)))))

def relu(x):
  return np.maximum(0, x)

def d_relu(x):
  return np.where(x > 0, 1, 0)

# funzione di costo

def cost_func(Ypred, Yr):

  Ypred = Ypred.astype(np.float64)
  Yr = Yr.astype(np.float64)
  eps = 10e-5
  Ypred = np.clip(Ypred, eps, 1-eps)

  return np.mean(-(Yr*np.log((Ypred))+(1-Yr)*np.log((1-Ypred))))

def d_cost_func(Ypred, Yr):

  Ypred = Ypred.astype(np.float64)
  Yr = Yr.astype(np.float64)
  eps = 10e-5
  Ypred = np.clip(Ypred, eps, 1-eps)

  return -(Yr/Ypred)+((1-Yr)/(1-Ypred))

# Definiamo iperparametri e creiamo la rete:

structure = [15, 3, 2, 1]
learning_rate = 0.01
epochs = 200

neural_network = build_network(structure, relu, d_relu, sigmoid, d_sigmoid)

loss_train = []
loss_test = []
acc_train = []
acc_test = []

for i in range(epochs):

  np.random.shuffle(df_new) # Mescoliamo il set di dati

  lines_train = int(df_new.shape[0]*0.7) # Lo dividiamo in 70/30
  data_train = df_new[:lines_train]
  data_test = df_new[lines_train:df_new.shape[0]]

  Y_train = data_train[:, -1]
  X_train = stand_matrix(np.array(data_train[:, :-1], dtype=np.float64))
  Y_test = data_test[:, -1]
  X_test = stand_matrix(np.array(data_test[:, :-1], dtype=np.float64))


  Y_train = Y_train[:, np.newaxis] # np.shape(Y_train) = (rows, 1)
  Y_test = Y_test[:, np.newaxis]

  # Alleniamo la rete e validiamo le sue predizioni

  pY_train = train_network(neural_network, X_train, Y_train, cost_func, d_cost_func, learning_rate)
  loss_train.append(cost_func(pY_train, Y_train))
  pY_test = train_network(neural_network, X_test, Y_test, cost_func, d_cost_func, learning_rate, mode_train=False)
  loss_test.append(cost_func(pY_test, Y_test))

  acc_train.append(precision(pY_train, Y_train))
  acc_test.append(precision(pY_test, Y_test))

  print(f'\rEpoca: {i}/{epochs}', end='', flush=True)



plt.figure(figsize=(14, 4))
clear_output(wait=True)
plt.subplot(1, 2, 1)
plt.plot(range(len(loss_train)), loss_train, label='Errore train')
plt.plot(range(len(loss_test)), loss_test, label='Errore test')
plt.title('Errore nel train e nei test')
plt.legend()
plt.xlabel('Epoche')
plt.ylabel('Errore')

plt.subplot(1, 2, 2)
plt.plot(range(len(acc_train)), acc_train, label='Error train')
plt.plot(range(len(acc_test)), acc_test, label='Error test')
plt.title('Precisione nel train e nei test')
plt.legend()
plt.xlabel('Epoche')
plt.ylabel('Precisione [%]')
plt.show()


end_execution = time.time()
print('-------------------------------------------------------------------------')
print('Network Configuration:')
print('Structure: ', structure)
print('Learning rate: ', learning_rate)
print('Epochs: ', epochs)
print('-------------------------------------------------------------------------')
print('Average error_train: ', np.mean(loss_train))
print('Average error_test: ', np.mean(loss_test))
print('-------------------------------------------------------------------------')
print('Average accuracy_train: ', np.mean(acc_train), '%')
print('Average accuracy_test: ', np.mean(acc_test), '%')
print('-------------------------------------------------------------------------')
print('Final error_train: ', loss_train[-1])
print('Final error_test: ', loss_test[-1])
print('-------------------------------------------------------------------------')
print('Final accuracy_train: ', acc_train[-1], '%')
print('Final accuracy_test: ', acc_test[-1], '%')
print('-------------------------------------------------------------------------')
print('Execution time: ', end_execution - start_execution)
print('-------------------------------------------------------------------------')