from neurona import *

class red_neuronal(Neurona):
    def __init__ (self, observaciones_en_entradas, pesos, peso_sesgo ):
        super().__init__(Neurona)
        super().__init__()



    def red_neuronas_multicapa(self, observaciones_en_entradas, pesos, peso_sesgo):

        #Cálculo de la activación de la primera capa
        primera_activacion = tf.sigmoid(tf.matmul(tf_neuronas_entradas_X, pesos['capa_entrada_hacia_oculta']) + peso_sesgo['peso_sesgo_capa_entrada_hacia_oculta'])

        #Cálculo de la activación de la segunda capa
        activacion_capa_oculta = tf.sigmoid(tf.matmul(primera_activacion, pesos['capa_oculta_hacia_salida']) + peso_sesgo['peso_sesgo_capa_oculta_hacia_salida'])

        return activacion_capa_oculta
