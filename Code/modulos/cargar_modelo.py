import tensorflow as tf

def cargar_modelo():
    modelo = tf.keras.models.load_model('\..\model\saved_model.pb')
    return modelo
