import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Neurona():
    def __init__ (self):
        self.observaciones = pd.read_csv("código cap11/datas/sonar.all-data.csv")
        self.epochs = 600
        self.cantidad_neuronas_entrada = 60
        self.cantidad_neuronas_salida = 2
        self.tasa_aprendizaje = 0.01

    def preparacion_datos(self):
        print("N.º columnas: ",len(self.observaciones.columns))

        #Para el aprendizaje solo tomamos loa datos procedentes del sonar
        X = self.observaciones[self.observaciones.columns[0:60]].values

        #Solo se toman los etiquetados
        y = self.observaciones[self.observaciones.columns[60]]

        #Se codifica: Las minas son iguales a 0 y las rocas son iguales 1
        encoder = LabelEncoder()
        encoder.fit(y)
        y = encoder.transform(y)

        n_labels = len(y)
        n_unique_labels = len(np.unique(y))
        one_hot_encode = np.zeros((n_labels,n_unique_labels))
        one_hot_encode[np.arange(n_labels),y] = 1
        Y=one_hot_encode


        #Verificación tomando los registros 0 y 97
        print("Clase Roca:",int(Y[0][1]))
        print("Clase Mina:",int(Y[97][1]))

        return X, Y

    def aprendizaje(self, X, Y):
        #Mezclamos
        X, Y = shuffle(X, Y, random_state=1)

        #Creación de los conjuntos de aprendizaje
        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.07, random_state=42)
        return train_x, test_x, train_y, test_y

    def parametrización(self, x):
        tf_neuronas_entradas_X = tf.placeholder(tf.float32,[None, 60])
        tf_valores_reales_Y = tf.placeholder(tf.float32,[None, 2])
        pesos = {
            #60 neuronas de las entradas hacia 24 Neuronas de la capa oculta
            'capa_entrada_hacia_oculta': tf.Variable(tf.random_uniform([60, x], minval=-0.3, maxval=0.3), tf.float32),

            # 12 neuronas de la capa oculta hacia 2 de la capa de salida
            'capa_oculta_hacia_salida': tf.Variable(tf.random_uniform([x, 2], minval=-0.3, maxval=0.3), tf.float32),


        }

        peso_sesgo = {
            #1 sesgo de la capa de entrada hacia las 24 neuronas de la capa oculta
            'peso_sesgo_capa_entrada_hacia_oculta': tf.Variable(tf.zeros([x]), tf.float32),

            #1 sesgo de la capa oculta hacia las 2 neuronas de la capa de salida
            'peso_sesgo_capa_oculta_hacia_salida': tf.Variable(tf.zeros([2]), tf.float32),
        }
        return tf_neuronas_entradas_X, tf_valores_reales_Y, pesos, peso_sesgo
    def red_neuronas_multicapa(self, tf_neuronas_entradas_X, pesos, peso_sesgo):

        #Cálculo de la activación de la primera capa
        primera_activacion = tf.sigmoid(tf.matmul(tf_neuronas_entradas_X, pesos['capa_entrada_hacia_oculta']) + peso_sesgo['peso_sesgo_capa_entrada_hacia_oculta'])

        #Cálculo de la activación de la segunda capa
        activacion_capa_oculta = tf.sigmoid(tf.matmul(primera_activacion, pesos['capa_oculta_hacia_salida']) + peso_sesgo['peso_sesgo_capa_oculta_hacia_salida'])

        return activacion_capa_oculta

    def error_optimizacion (self, tf_neuronas_entradas_X, tf_valores_reales_Y, pesos, peso_sesgo):
        red = Neurona.red_neuronas_multicapa(self, tf_neuronas_entradas_X, pesos, peso_sesgo)

        funcion_error = tf.reduce_sum(tf.pow(tf_valores_reales_Y-red,2))
        funcion_precision = tf.metrics.accuracy(labels=tf_valores_reales_Y,predictions=red)
        optimizador = tf.train.GradientDescentOptimizer(learning_rate=self.tasa_aprendizaje).minimize(funcion_error)
        return red, funcion_error, funcion_precision, optimizador


    def aprendizaje2(self, funcion_error, funcion_precision, optimizador, tf_neuronas_entradas_X, tf_valores_reales_Y, train_x, train_y):
        init = tf.global_variables_initializer()
        #Inicio de una sesión de aprendizaje
        sesion = tf.Session()
        sesion.run(init)

        #Para la realización de la gráfica para la MSE
        Grafica_MSE=[]


        #Para cada epoch
        for i in range(self.epochs):

            #Realización del aprendizaje con actualización de los pesos
            sesion.run(optimizador, feed_dict = {tf_neuronas_entradas_X: train_x, tf_valores_reales_Y: train_y})

            #Calcular el error de aprendizaje
            MSE = sesion.run(funcion_error, feed_dict = {tf_neuronas_entradas_X: train_x, tf_valores_reales_Y: train_y})

            #Visualización de la información
            Grafica_MSE.append(MSE)
            print("EPOCH (" + str(i) + "/" + str(self.epochs) + ") -  MSE: "+ str(MSE))
        return Grafica_MSE, sesion

    def plot(self, graf):
        plt.plot(graf)
        plt.ylabel('MSE')
        plt.show()

    def verf_aprendizaje(self, red, argmaxim, tf_valores_reales_Y):
        clasificaciones = tf.argmax(red, argmaxim)
        formula_calculo_clasificaciones_correctas = tf.equal(clasificaciones, tf.argmax(tf_valores_reales_Y,1))
        formula_precision = tf.reduce_mean(tf.cast(formula_calculo_clasificaciones_correctas, tf.float32))
        return clasificaciones, formula_precision

    def precision_pruebas (self, sesion, clasificaciones, formula_precision, tf_neuronas_entradas_X, tf_valores_reales_Y, test_x, test_y):
        n_clasificaciones = 0;
        n_clasificaciones_correctas = 0

        #Se mira el conjunto de los datos de prueba (text_x)
        for i in range(0,test_x.shape[0]):

            #Se recupera la información
            datosSonar = test_x[i].reshape(1,60)
            clasificacionEsperada = test_y[i].reshape(1,2)

            # Se realiza la clasificación
            prediccion_run = sesion.run(clasificaciones, feed_dict={tf_neuronas_entradas_X:datosSonar})

            #Se calcula la precisión de la clasificación con la ayuda de la fórmula establecida antes
            accuracy_run = sesion.run(formula_precision, feed_dict={tf_neuronas_entradas_X:datosSonar, tf_valores_reales_Y:clasificacionEsperada})


            #Se muestra para observación la clase original y la clasificación realizada
            print(i,"Clase esperada: ", int(sesion.run(tf_valores_reales_Y[i][1],feed_dict={tf_valores_reales_Y:test_y})), "Clasificación: ", prediccion_run[0] )

            n_clasificaciones = n_clasificaciones+1
            if(accuracy_run*100 ==100):
                n_clasificaciones_correctas = n_clasificaciones_correctas+1


        print("-------------")
        print("Precisión en los datos de pruebas = "+str((n_clasificaciones_correctas/n_clasificaciones)*100)+"%")


    def precision_aprendizaje(self,  sesion, clasificaciones, formula_precision, train_x, train_y, tf_neuronas_entradas_X, tf_valores_reales_Y):
        n_clasificaciones = 0;
        n_clasificaciones_correctas = 0
        for i in range(0, train_x.shape[0]):

            # Recuperamos la información
            datosSonar = train_x[i].reshape(1, 60)
            clasificacionEsperada = train_y[i].reshape(1, 2)

            # Realizamos la clasificación
            prediccion_run = sesion.run(clasificaciones, feed_dict={tf_neuronas_entradas_X: datosSonar})

            # Calculamos la precisión de la clasificación con la ayuda de la fórmula establecida antes
            accuracy_run = sesion.run(formula_precision, feed_dict={tf_neuronas_entradas_X: datosSonar, tf_valores_reales_Y: clasificacionEsperada})

            n_clasificaciones = n_clasificaciones + 1
            if (accuracy_run * 100 == 100):
                n_clasificaciones_correctas = n_clasificaciones_correctas + 1


        print("Precisión en los datos de aprendizaje = " + str((n_clasificaciones_correctas / n_clasificaciones) * 100) + "%")


    def precision_datos(self, sesion, clasificaciones, formula_precision, tf_neuronas_entradas_X, tf_valores_reales_Y, X, Y):
        n_clasificaciones = 0;
        n_clasificaciones_correctas = 0
        for i in range(0,207):

            prediccion_run = sesion.run(clasificaciones, feed_dict={tf_neuronas_entradas_X:X[i].reshape(1,60)})
            accuracy_run = sesion.run(formula_precision, feed_dict={tf_neuronas_entradas_X:X[i].reshape(1,60), tf_valores_reales_Y:Y[i].reshape(1,2)})

            n_clasificaciones = n_clasificaciones + 1
            if (accuracy_run * 100 == 100):
                n_clasificaciones_correctas = n_clasificaciones_correctas + 1


        print("Precisión en el conjunto de datos = " + str((n_clasificaciones_correctas / n_clasificaciones) * 100) + "%")




        sesion.close()





def main ():
    neurona = Neurona()
    X, Y = neurona.preparacion_datos()
    train_x, test_x, train_y, test_y = neurona.aprendizaje(X, Y)
    tf_neuronas_entradas_X, tf_valores_reales_Y, pesos, peso_sesgo =  neurona.parametrización(26)
    #Creación de la red neuronal
    neurona.red_neuronas_multicapa(tf_neuronas_entradas_X, pesos, peso_sesgo)
    red, funcion_error, funcion_precision, optimizador = neurona.error_optimizacion(tf_neuronas_entradas_X, tf_valores_reales_Y, pesos, peso_sesgo)
    grafica, sesion = neurona.aprendizaje2(funcion_error, funcion_precision, optimizador, tf_neuronas_entradas_X, tf_valores_reales_Y, train_x, train_y)
    neurona.plot(grafica)
    clasificaciones, formula_precision = neurona.verf_aprendizaje(red, 1, tf_valores_reales_Y)
    neurona.precision_pruebas(sesion, clasificaciones, formula_precision,  tf_neuronas_entradas_X, tf_valores_reales_Y, test_x, test_y)
    neurona.precision_aprendizaje(sesion, clasificaciones, formula_precision, train_x, train_y, tf_neuronas_entradas_X, tf_valores_reales_Y)
    neurona.precision_datos(sesion, clasificaciones, formula_precision, tf_neuronas_entradas_X, tf_valores_reales_Y, X, Y)


if __name__ == "__main__":
    main()