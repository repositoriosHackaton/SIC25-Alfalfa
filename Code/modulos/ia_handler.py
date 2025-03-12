import tensorflow as tf
import numpy as np

class IAHandler:
    def __init__(self):
        self.modelo = None

    def cargar_modelo(self):
        self.modelo = tf.keras.models.load_model('\..\model\saved_model.pb')

    def predecir(self, datos_entrada):
        datos = np.array([float(x) for x in datos_entrada.split(',')])
        prediccion = self.modelo.predict(datos.reshape(1, -1))
        return prediccion
