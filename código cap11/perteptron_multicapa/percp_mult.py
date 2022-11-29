import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import numpy as np

class Percepton():
    def __init__ (self, obs_entr, prediccion, n_neuronas_capa_oculta, epochs):
        self.valores_entradas_X = obs_entr
        self.valores_a_predecir_Y = prediccion
        self.tf_neuronas_entradas_X = tf.placeholder(tf.float32, [None, 2])
        self.tf_valores_reales_Y = tf.placeholder(tf.float32, [None, 1])
        self.n_neuronas_capa_oculta = n_neuronas_capa_oculta
        self.epochs = epochs


    def pesos(self):
        #Los primeros están 4 : 2 en la entrada (X1 y X2) y 2 pesos por entrada
        pesos = tf.Variable(tf.random_normal([2, 2]), tf.float32)
        #los pesos de la capa oculta están 2 : 2 en la entrada (H1 y H2) y 1 peso por entrada
        peso_capa_oculta = tf.Variable(tf.random_normal([2, 1]), tf.float32)
        return pesos, peso_capa_oculta

    del calculo_sesgo(self):
        #El primer sesgo contiene 2 pesos
        sesgo = tf.Variable(tf.zeros([2]))
        #El segundo sesgo contiene 1 peso
        sesgo_capa_oculta = tf.Variable(tf.zeros([1]))
        return sesgo, sesgo_capa_oculta

    def activacion(self, pesos, sesgo, peso_capa_oculta, sesgo_capa_oculta):
        #Cálculo de la activación de la primera capa
        #cálculo de la suma ponderada (tf.matmul) con ayuda de los datos X1, X2, W11, W12, W31, W41 y del sesgo
        #después aplicación de la función sigmoide (tf.sigmoid)
        activacion = tf.sigmoid(tf.matmul(self.tf_neuronas_entradas_X, pesos) + sesgo)

        #Cálculo de la activación de la capa oculta
        #cálculo de la suma ponderada (tf.matmul) con ayuda de los datos H1, H2, W12, W21 y del sesgo
        #después aplicación de la función sigmoide (tf.sigmoid)
        activacion_capa_oculta = tf.sigmoid(tf.matmul(activacion, peso_capa_oculta) + sesgo_capa_oculta)
        return activacion, activacion_capa_oculta

    def funcion_error(self, activacion_capa_oculta ):
        #Función de error de media cuadrática MSE
        funcion_error = tf.reduce_sum(tf.pow(self.tf_valores_reales_Y-activacion_capa_oculta,2))
        return funcion_error

    def optimizador (self, funcion_error):
        #Descenso del gradiente con una tasa de aprendizaje fijada en 0,1
        optimizador = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(funcion_error)
        return optimizador

    def aprendizaje(self):
        #Inicialización de la variable
        init = tf.global_variables_initializer()

        #Inicio de una sesión de aprendizaje
        sesion = tf.Session()
        sesion.run(init)

        #Para la realización de la gráfica para la MSE
        Grafica_MSE=[]


        #Para cada epoch
        for i in range(self.epochs):

            #Realización del aprendizaje con actualización de los pesos
            sesion.run(optimizador, feed_dict = {self.tf_neuronas_entradas_X: self.valores_entradas_X, self.tf_valores_reales_Y: self.valores_a_predecir_Y})

            #Calcular el error
            MSE = sesion.run(funcion_error, feed_dict = {self.tf_neuronas_entradas_X: self.valores_entradas_X, self.tf_valores_reales_Y: self.valores_a_predecir_Y})

            #Visualización de la información
            Grafica_MSE.append(MSE)
            print("EPOCH (" + str(i) + "/" + str(epochs) + ") -  MSE: "+ str(MSE))
        return graf, sesion

    def grafica(self, Grafica_MSE):
        #Visualización gráfica
        import matplotlib.pyplot as plt
        plt.plot(Grafica_MSE)
        plt.ylabel('MSE')
        plt.show()


    def verificaciones(self, sesion, activacion_capa_oculta,):
        print("--- VERIFICACIONES ----")

        for i in range(0,4):
            print("Observación:"+str(self.valores_entradas_X[i])+ " - Esperado: "+str(self.valores_a_predecir_Y[i])+" - Predicción: "+str(sesion.run(activacion_capa_oculta, feed_dict={self.tf_neuronas_entradas_X: [self.valores_entradas_X[i]]})))



        sesion.close()



def main():
    perceptron = Percepton([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], [[0.], [1.], [1.], [0.]], 2, 1000)
    pesos, peso_capa_oculta = perceptron.pesos()
    sesgo, sesgo_capa_oculta = perceptron.calculo_sesgo()
    activacion,  activacion_capa_oculta = perceptron.activacion( pesos, sesgo, peso_capa_oculta, sesgo_capa_oculta)
    funcion_error = funcion_error(activacion_capa_oculta)
    optimizador = optimizador(funcion_error)
