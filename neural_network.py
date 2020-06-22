import numpy as np

#red de 14 entadas, capa oculta de 8 nodos y una nodo de salida
class NeuralNet():

    def __init__(self, capas=[14, 7, 1], ratio_aprendizaje=0.001, iterations=4100):
        self.parametros = {} #aca se guardaran los pesos y bias
        self.capas = capas
        self.ratio_aprendizaje = ratio_aprendizaje
        self.iterations = iterations
        # self.perdida = []
        self.X = None
        self.y = None

    def init_pesos(self):
        # inicializar los pesos de una distribucion normal aleatoria

        np.random.seed(1)  # inicializamos semilla aleatoria(número utilizado para inicializar un generador de números pseudoaleatorios)
        self.parametros["W1"] = np.random.randn(self.capas[0], self.capas[1])# esta matriz de peso tendra un peso de 14 x 7
        self.parametros['b1'] = np.random.randn(self.capas[1], ) # primero bias sera un vector de tamaño 7 porque tiene 7 nodos ocultos
        self.parametros['W2'] = np.random.randn(self.capas[1], self.capas[2]) # la segunda matriz de pesos sera de 7 x 1 porque tiene 7 nodos ocultos y un nodo de salida
        self.parametros['b2'] = np.random.randn(self.capas[2], ) # solo un tamño porque solo hay ua salida

    def relu(self, Z):

        # El relu realiza una operacion umbral a cada elemento donde los valores menores que 0 se establecen en 0
        #esto realiza un relu para matrices porque principalmente estara lidiando con arrays y no valores unicos
        return np.maximum(0, Z)

    def sigmoide(self, Z):
        #La función sigmoide toma números reales en cualquier rango y los comprime a una salida de valor real entre 0 y 1.

        return 1.0 / (1.0 + np.exp(-Z))

    def entropia_cruzada(self, y, yhat):
        nsample = len(y)
        perdida = -1 / nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((1 - y), np.log(1 - yhat + 0.000001))))
        return perdida

    def forward_propagation(self):

        #Realiza la propagación hacia adelante.

        Z1 = self.X.dot(self.parametros['W1']) + self.parametros['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.parametros['W2']) + self.parametros['b2']
        y_calculada = self.sigmoide(Z2)
        perdida = self.entropia_cruzada(self.y, y_calculada)

        # guardar los valores calcualdos
        self.parametros['Z1'] = Z1
        self.parametros['Z2'] = Z2
        self.parametros['A1'] = A1

        return y_calculada, perdida

    def back_propagation(self, y_calculada):
        #Calcula los derivados y actualiza los pesos y bias.

        def dRelu(x):
            x[x <= 0] = 0
            x[x > 0] = 1
            return x

        # calculamos las derivadas
        derivada_perdida_respecto_a_y_calculada = -(np.divide(self.y, y_calculada) - np.divide((1 - self.y), (1 - y_calculada + 0.000001)))
        derivada_perdida_respecto_a_sig = y_calculada * (1 - y_calculada)
        derivada_perdida_respecto_a_z2 = derivada_perdida_respecto_a_y_calculada * derivada_perdida_respecto_a_sig

        derivada_perdida_respecto_a_A1 = derivada_perdida_respecto_a_z2.dot(self.parametros['W2'].T)
        derivada_perdida_respecto_a_w2 = self.parametros['A1'].T.dot(derivada_perdida_respecto_a_z2)
        derivada_perdida_respecto_a_b2 = np.sum(derivada_perdida_respecto_a_z2, axis=0)

        derivada_perdida_respecto_a_z1 = derivada_perdida_respecto_a_A1 * dRelu(self.parametros['Z1'])
        derivada_perdida_respecto_a_w1 = self.X.T.dot(derivada_perdida_respecto_a_z1)
        derivada_perdida_respecto_a_b1 = np.sum(derivada_perdida_respecto_a_z1, axis=0)

        # actualizar los pesos y bias
        self.parametros['W1'] = self.parametros['W1'] - self.ratio_aprendizaje * derivada_perdida_respecto_a_w1
        self.parametros['W2'] = self.parametros['W2'] - self.ratio_aprendizaje * derivada_perdida_respecto_a_w2
        self.parametros['b1'] = self.parametros['b1'] - self.ratio_aprendizaje * derivada_perdida_respecto_a_b1
        self.parametros['b2'] = self.parametros['b2'] - self.ratio_aprendizaje * derivada_perdida_respecto_a_b2

    def entrenar(self, X, y):
        #Entrena la red neuronal usuando la data y etiquetas

        self.X = X
        self.y = y
        self.init_pesos()  # inicializar pesos y bias

        for i in range(self.iterations):
            y_calculada, perdida = self.forward_propagation()
            self.back_propagation(y_calculada)

            # self.perdida.append(perdida)

    def predecir(self, X):

        Z1 = X.dot(self.parametros['W1']) + self.parametros['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.parametros['W2']) + self.parametros['b2']
        prediccion = self.sigmoide(Z2)
        # redondear la prediccion
        return np.round(prediccion)

    def exactitud(self, y, y_calculada):
        #calcular exactitud de los resulatdos calculado con las etiquetas reales

        acc = int(sum(y == y_calculada) / len(y) * 100)
        return acc