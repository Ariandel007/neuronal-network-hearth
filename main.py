import pandas as pd
import neural_network
from sklearn.preprocessing import StandardScaler

#leer archivos
arch = pd.read_csv('dataset.csv', sep=';')
#Columnas de dataset

#quitar la columna diagnosis y qrs
X = arch.drop(columns=['diagnosis'])

# reemplazaremos el diagnostico con 0 y 1 (0 no esta enfermo, 1 esta enfermo)
arch['diagnosis'] = arch['diagnosis'].replace(1, 0)
arch['diagnosis'] = arch['diagnosis'].replace(2, 1)

y_label = arch['diagnosis'].values.reshape(X.shape[0], 1)

# estandarizar el dataset
sc = StandardScaler()
sc.fit(X)

X = sc.transform(X)
nn = neural_network.NeuralNet()
nn.entrenar(X, y_label)

#predecir
result = nn.predecir(X)
print(result)

precistion = nn.exactitud(y_label, result)

# verificar que se paresca
print("procentaje de precision", float(precistion), "%")
result = nn.predecir(X)

test = [[70, 1, 160, 84, 100, 165, 397, 163, 108, -10, 88, 0, 48, 32]]

test_nn =sc.transform(test)

print(nn.predecir(test_nn))

nn.plot_perdida()