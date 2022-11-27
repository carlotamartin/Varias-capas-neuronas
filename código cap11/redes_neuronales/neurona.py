import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Neurona():
    def __init__ (self, reader, epochs,cantidad_neuronas_entrada,  cantidad_neuronas_salida, tasa_aprendizaje):
        self.observaciones = pd.read_csv(reader)
        self.epochs = epochs
        self.cantidad_neuronas_entrada = cantidad_neuronas_entrada
        self.cantidad_neuronas_salida = cantidad_neuronas_salida
        self.tasa_aprendizaje = tasa_aprendizaje

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

        return Y

    def aprendizaje(self):
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




def main ():
    neurona = Neurona("código cap11/datas/sonar.all-data.csv")
    neurona.preparacion_datos()
    neurona.aprendizaje()
    tf_neuronas_entradas_X, tf_valores_reales_Y =  neurona.parametrización(12)
if __name__ == "__main__":
    main()