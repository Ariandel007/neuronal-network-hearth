import numpy as np
import pandas as pd

class NeuralNetwork:

    def __init__(self, init_inputs, salidasEsperadas, const_aprendizaje, neuronasCapaEntrada, neuronasCapaOculta, neuronasCapaSalida):
        self.init_inputs = init_inputs
        self.salidasEsperada = salidasEsperadas

        self.const_aprendizaje = const_aprendizaje
        self.neuronasCapaEntrada = neuronasCapaEntrada
        self.neuronasCapaOculta = neuronasCapaOculta
        self.neuronasCapaSalida = neuronasCapaSalida


        self.pesos_capaOculta = np.random.uniform(size=(self.neuronasCapaEntrada, self.neuronasCapaOculta))
        self.bias_capaOculta = np.random.uniform(size=(1, self.neuronasCapaOculta))
        self.pesos_capaSalida = np.random.uniform(size=(self.neuronasCapaOculta, self.neuronasCapaSalida))
        self.bias_capaSalida = np.random.uniform(size=(1, self.neuronasCapaSalida))
        self.hidden_layer_output = float
        self.errors = []
        self.output = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def mostar_datos_iniciales(self):
        print("Pesos iniciales de la capa oculta: ", end='')
        print(*self.pesos_capaOculta)
        print("BIAS inicial de la capa oculta: ", end='')
        print(*self.bias_capaOculta)
        print("Pesos iniciales de la capa de salida: ", end='')
        print(*self.pesos_capaSalida)
        print("BIAS inicial de la capa de salida: ", end='')
        print(*self.bias_capaSalida)
        print("///////////////////////////////////////////////////////////////////////////////////////////////////////")

    def forward_propagation(self, inputs):
        hidden_layer_activation = np.dot(inputs, self.pesos_capaOculta)
        hidden_layer_activation += self.bias_capaOculta
        self.hidden_layer_output = self.sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(self.hidden_layer_output, self.pesos_capaSalida)
        output_layer_activation += self.bias_capaSalida
        predicted_output = self.sigmoid(output_layer_activation)

        return predicted_output

    def entrenar(self, num_iterations):
        for _ in range(num_iterations):

            self.output = self.forward_propagation(self.init_inputs)

            # Backpropagation
            error = self.salidasEsperada - self.output
            d_predicted_output = error * self.sigmoid_derivative(self.output)
            self.errors.append(error)

            error_hidden_layer = d_predicted_output.dot(self.pesos_capaSalida.T)
            d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(self.hidden_layer_output)

            # Actualizar los pesos y Biases
            self.pesos_capaSalida += self.hidden_layer_output.T.dot(d_predicted_output) * self.const_aprendizaje
            self.bias_capaSalida += np.sum(d_predicted_output) * self.const_aprendizaje
            self.pesos_capaOculta += self.init_inputs.T.dot(d_hidden_layer) * self.const_aprendizaje
            self.bias_capaOculta += np.sum(d_hidden_layer) * self.const_aprendizaje

    def mostrar_datos_finales(self):
        print("Peso finales de la capa oculta: ", end='')
        print(self.pesos_capaOculta)
        print("BIAS finales de la capa oculta: ", end='')
        print(self.bias_capaOculta)
        print("Peso finales de la capa salida:  ", end='')
        print(self.pesos_capaSalida)
        print("BIAS finales de la capa salida: ", end='')
        print(self.bias_capaSalida)
        print("\nSalidas de la red neuronal luego de 10,000 epocas: ", end='   \n')
        print(self.output)


#leer archivos
arch = pd.read_csv('dataset.csv', sep=';')
#Columnas de dataset

age = arch['age'].values
sex = arch['sex'].values
height = arch['height'].values
weight = arch['weight'].values
qrs_duration = arch['qrs_duration'].values
pr_interval = arch['p-r_interval'].values
qt_interval = arch['q-t_interval'].values
t_interval = arch['t_interval'].values
p_interval = arch['p_interval'].values
qrs = arch['qrs'].values
heart_rate = arch['heart_rate'].values
q_wave = arch['q_wave'].values
r_wave = arch['r_wave'].values
s_wave = arch['s_wave'].values

diagnosis = arch['diagnosis'].values

#numero de pesos
num_w = 14
#numero de pesos capa oculta
num_w_h = 7
#entradas
arr_input = []
#etiquetas
arr_output = []

size_dataset = len(age)

for i in range(size_dataset):
    arr_elements = []
    arr_elements.append(int(age[i]))
    arr_elements.append(int(sex[i]))
    arr_elements.append(int(height[i]))
    arr_elements.append(int(weight[i]))
    arr_elements.append(int(qrs_duration[i]))
    arr_elements.append(int(pr_interval[i]))
    arr_elements.append(int(qt_interval[i]))
    arr_elements.append(int(t_interval[i]))
    arr_elements.append(int(p_interval[i]))
    arr_elements.append(int(qrs[i]))
    if heart_rate[i] == '?':
        heart_rate[i] = 72
    arr_elements.append(int(heart_rate[i]))
    arr_elements.append(int(q_wave[i]))
    arr_elements.append(int(r_wave[i]))
    arr_elements.append(int(s_wave[i]))


    arr_input.append(arr_elements)

    if diagnosis[i] == 2:
        arr_out = [0]
    else:
        arr_out = [1]
    arr_output.append(arr_out)


inputs = np.array(arr_input)

expected_output = np.array(arr_output)

nn = NeuralNetwork(inputs, expected_output, 0.1, num_w, num_w_h, 1)

nn.mostar_datos_iniciales()
print("-------------------------------------")
nn.entrenar(20000)
print("-------------------------------------")
nn.mostrar_datos_finales()
# print("-----------probemoslo--------------------------")
# #si sale 1 esta sano, si sale 0 esta enfermo
# num = nn.forward_propagation(np.array([[54, 1, 160, 63, 82, 158, 410, 141, 87, 25, 54, 0, 48, 0]]))[0][0]
# print(num)
# print(round(num))
