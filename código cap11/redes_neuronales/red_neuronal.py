from neurona import *
import matplotlib.pyplot as plt


class red_neuronal(Neurona):
    def __init__ (self, tasa_de_aprendizaje, epochs):
        super().__init__(Neurona)
        super().__init__(epochs= epochs)
        super().__init__(tasa_aprendizaje= tasa_de_aprendizaje)
        self.Y = Neurona.preparacion_datos()
        self.train_x, self.test_x, self.train_y, self.test_y  = Neurona.aprendizaje()
        self.tf_neuronas_entradas_X, self.tf_valores_reales_Y, self.pesos, self.peso_sesgo = Neurona.parametrización()

    def red_neuronas_multicapa(self):

        #Cálculo de la activación de la primera capa
        primera_activacion = tf.sigmoid(tf.matmul(self.tf_neuronas_entradas_X, self.pesos['capa_entrada_hacia_oculta']) + self.peso_sesgo['peso_sesgo_capa_entrada_hacia_oculta'])

        #Cálculo de la activación de la segunda capa
        activacion_capa_oculta = tf.sigmoid(tf.matmul(primera_activacion, self.pesos['capa_oculta_hacia_salida']) + self.peso_sesgo['peso_sesgo_capa_oculta_hacia_salida'])

        return activacion_capa_oculta

    def error_optimizacion (self):
        red = red_neuronal.red_neuronas_multicapa(self.tf_neuronas_entradas_X, self.pesos, self.peso_sesgo)

        funcion_error = tf.reduce_sum(tf.pow(self.tf_valores_reales_Y-red,2))
        funcion_precision = tf.metrics.accuracy(labels=self.tf_valores_reales_Y,predictions=red)
        optimizador = tf.train.GradientDescentOptimizer(learning_rate=self.tasa_aprendizaje).minimize(funcion_error)
        return funcion_error, funcion_precision, optimizador


    def aprendizaje(self, funcion_error, funcion_precision, optimizador):
        init = tf.global_variables_initializer()
        #Inicio de una sesión de aprendizaje
        sesion = tf.Session()
        sesion.run(init)

        #Para la realización de la gráfica para la MSE
        Grafica_MSE=[]


        #Para cada epoch
        for i in range(self.epochs):

            #Realización del aprendizaje con actualización de los pesos
            sesion.run(optimizador, feed_dict = {self.tf_neuronas_entradas_X: self.train_x, self.tf_valores_reales_Y:self.train_y})

            #Calcular el error de aprendizaje
            MSE = sesion.run(funcion_error, feed_dict = {self.tf_neuronas_entradas_X: self.train_x, self.tf_valores_reales_Y:self.train_y})

            #Visualización de la información
            Grafica_MSE.append(MSE)
            print("EPOCH (" + str(i) + "/" + str(self.epochs) + ") -  MSE: "+ str(MSE))
        return Grafica_MSE, sesion

    def plot(self, graf):
        plt.plot(graf)
        plt.ylabel('MSE')
        plt.show()

    def verf_aprendizaje(self, red, argmaxim):
        clasificaciones = tf.argmax(red, argmaxim)
        formula_calculo_clasificaciones_correctas = tf.equal(clasificaciones, tf.argmax(self.tf_valores_reales_Y,1))
        formula_precision = tf.reduce_mean(tf.cast(formula_calculo_clasificaciones_correctas, tf.float32))
        return clasificaciones, formula_precision

    def precision_pruebas (self, sesion, clasificaciones, formula_precision):
        n_clasificaciones = 0;
        n_clasificaciones_correctas = 0

        #Se mira el conjunto de los datos de prueba (text_x)
        for i in range(0,self.test_x.shape[0]):

            #Se recupera la información
            datosSonar = self.test_x[i].reshape(1,60)
            clasificacionEsperada = self.test_y[i].reshape(1,2)

            # Se realiza la clasificación
            prediccion_run = sesion.run(clasificaciones, feed_dict={self.tf_neuronas_entradas_X:datosSonar})

            #Se calcula la precisión de la clasificación con la ayuda de la fórmula establecida antes
            accuracy_run = sesion.run(formula_precision, feed_dict={self.tf_neuronas_entradas_X:datosSonar, self.tf_valores_reales_Y:clasificacionEsperada})


            #Se muestra para observación la clase original y la clasificación realizada
            print(i,"Clase esperada: ", int(sesion.run(self.tf_valores_reales_Y[i][1],feed_dict={self.tf_valores_reales_Y:self.test_y})), "Clasificación: ", prediccion_run[0] )

            n_clasificaciones = n_clasificaciones+1
            if(accuracy_run*100 ==100):
                n_clasificaciones_correctas = n_clasificaciones_correctas+1


        print("-------------")
        print("Precisión en los datos de pruebas = "+str((n_clasificaciones_correctas/n_clasificaciones)*100)+"%")


    def precision_aprendizaje(self,  sesion, clasificaciones, formula_precision):
        n_clasificaciones = 0;
        n_clasificaciones_correctas = 0
        for i in range(0,self.train_x.shape[0]):

            # Recuperamos la información
            datosSonar = self.train_x[i].reshape(1, 60)
            clasificacionEsperada = self.train_y[i].reshape(1, 2)

            # Realizamos la clasificación
            prediccion_run = sesion.run(clasificaciones, feed_dict={self.tf_neuronas_entradas_X: datosSonar})

            # Calculamos la precisión de la clasificación con la ayuda de la fórmula establecida antes
            accuracy_run = sesion.run(formula_precision, feed_dict={self.tf_neuronas_entradas_X: datosSonar, self.tf_valores_reales_Y: clasificacionEsperada})

            n_clasificaciones = n_clasificaciones + 1
            if (accuracy_run * 100 == 100):
                n_clasificaciones_correctas = n_clasificaciones_correctas + 1


        print("Precisión en los datos de aprendizaje = " + str((n_clasificaciones_correctas / n_clasificaciones) * 100) + "%")


    def precision_datos(self, sesion, clasificaciones, formula_precision):
        n_clasificaciones = 0;
        n_clasificaciones_correctas = 0
        for i in range(0,207):

            prediccion_run = sesion.run(clasificaciones, feed_dict={self.tf_neuronas_entradas_X:X[i].reshape(1,60)})
            accuracy_run = sesion.run(formula_precision, feed_dict={self.tf_neuronas_entradas_X:X[i].reshape(1,60), self.tf_valores_reales_Y:self.Y[i].reshape(1,2)})

            n_clasificaciones = n_clasificaciones + 1
            if (accuracy_run * 100 == 100):
                n_clasificaciones_correctas = n_clasificaciones_correctas + 1


        print("Precisión en el conjunto de datos = " + str((n_clasificaciones_correctas / n_clasificaciones) * 100) + "%")




        sesion.close()

def main ():
    red = red_neuronal()
    red.red_neuronas_multicapa()
    funcion_error, funcion_precision, optimizador = red.error_optimizacion()
    grafica, sesion = red.aprendizaje(funcion_error, funcion_precision, optimizador)
    red.plot(grafica)
    clasificaciones, formula_precision = red.verf_aprendizaje(red, 1)
    red.precision_pruebas(sesion, clasificaciones, formula_precision)
    red.precision_aprendizaje(sesion, clasificaciones, formula_precision)
    red.precision_datos(sesion, clasificaciones, formula_precision)


