import sys
import tensorflow as tf
import numpy as np
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

texto = "Este es un ejemplo de cómo tokenizar un texto usando BERT."

tokens = tokenizer.tokenize(texto)

tokens_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Tokens:", tokens)
print("IDs de Tokens:", tokens_ids)

texto_decodificado = tokenizer.decode(tokens_ids)

print("Texto Decodificado:", texto_decodificado)

# Argumentos de la línea de comando (datos de entrada)
datos_entrada = sys.argv[1]

# Convertir los datos de entrada a un formato adecuado
datos = np.array([float(x) for x in datos_entrada.split(',')])

# Cargar el modelo
modelo = tf.keras.models.load_model('\..\model\saved_model.pb')

# Hacer la predicción
prediccion = modelo.predict(datos.reshape(1, -1))

print(prediccion)
