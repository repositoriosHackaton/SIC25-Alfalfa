# Este modulo tenia la intencion de funcionar como la conjuncion entre el modelo BERt preentrenado
# y la CNN, no dio los resultados esperados y estamos trabajando en notebook2.ipynb exclusivamente 
# con el modelo BERT
import pandas as pd
import re
import joblib
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

nltk.download('stopwords')
nltk.download('wordnet')

# preprocesado
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    stop_words = set(stopwords.words('english'))  # English stopwords
    words = [word for word in text.split() if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# preprocesado
def preprocess_data(df, text_column='text'):
    df[text_column] = df[text_column].apply(preprocess_text)
    return df

# binarizar 
def binarize(df,label_column = 'label'):
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(df[label_column])
    labels_df = pd.DataFrame(labels, columns=mlb.classes_)
    return labels_df, mlb

# tokenizar
def tokenize_data_in_batches(tokenizer, texts, max_length=128, batch_size=32):
    encodings = {'input_ids': [], 'attention_mask': []}
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_encodings = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors='tf'
        )
        encodings['input_ids'].append(batch_encodings['input_ids'])
        encodings['attention_mask'].append(batch_encodings['attention_mask'])
    encodings['input_ids'] = tf.concat(encodings['input_ids'], axis=0)
    encodings['attention_mask'] = tf.concat(encodings['attention_mask'], axis=0)
    return encodings

# crear dataset tf
def create_tf_datasets(train_encodings, y_train, val_encodings, y_test, batch_size=16):
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).shuffle(1000).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), y_test)).batch(batch_size)
    return train_dataset, val_dataset

# CARGAR MODELO PRE ENTRENADO Y TOKENIZADOR
def create_bert(model_name= 'bert-base-uncased'):
    model = TFBertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Dividir datos
def split_data(df, labels_df, test_size=0.2, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(
        df['text'], labels_df, test_size=test_size, random_state=random_state
    )
    return x_train, x_test, y_train, y_test

#OPCIONAL
def tune_bert(train_dataset, val_dataset, model_name='bert-base-uncased', num_labels=None, epochs=5):
    model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[early_stopping])
    return model

# CREAR RED CONVOLUCIONADA
def create_cnn(bert_model, num_labels):
    input_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.Input(shape=(None,), dtype=tf.int32, name='attention_mask')
    bert_output = bert_model(input_ids, attention_mask=attention_mask)
    x = bert_output.last_hidden_state

    x = tf.keras.layers.Conv1D(filters=128,kernel_size=3, activation = 'relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(filters = 64, kernel_size=3, activation = 'relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_labels, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=outputs)
    return model

# Entrenar BERT + CNN
def train_mixed_model(model,train_dataset,val_dataset, epochs= 10, patience = 3):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss',patience=patience)
    history = model.fit(train_dataset,validation_data = val_dataset, epochs=epochs, callbacks=[early_stopping])
    return history

# funcion del profe :)
def categorizar_articulo_subtemas_en(texto, palabras_clave, vectorizer, umbrales, top_n=3):
    try:
        texto = texto.lower()
        texto = re.sub(r'[^\w\s]', '', texto)  # remover puntuacion
        stop_words = set(stopwords.words('english'))  # English stopwords
        palabras = [palabra for palabra in texto.split() if palabra not in stop_words]

        # Lemmatizacion
        lemmatizer = WordNetLemmatizer()
        palabras = [lemmatizer.lemmatize(palabra) for palabra in palabras]

        texto_limpio = " ".join(palabras)  
        vector_texto = vectorizer.transform([texto_limpio])  # Convertir texto a vector

        
        similitudes = {}
        for subtema, palabras_clave_subtema in palabras_clave.items():
            vector_palabras_clave = vectorizer.transform([" ".join(palabras_clave_subtema)])
            similitudes[subtema] = cosine_similarity(vector_texto, vector_palabras_clave)[0][0]

        # organizar por similaridad
        categorias_ordenadas = sorted(similitudes.items(), key=lambda x: x[1], reverse=True)

        # seleccionar top categorias
        top_categorias = []
        for categoria, similitud in categorias_ordenadas:
            umbral_categoria = umbrales.get(categoria, 0.1)  
            if similitud >= umbral_categoria:
                top_categorias.append(categoria)

        if not top_categorias:
            top_categorias.append("No specific subtopic")

        return top_categorias[:top_n]  

    except Exception as e:
        print(f"Error al procesar el texto: {e}")
        return ["Error"]  
    
def categorizar(df):
    subtemas_palabras_clave, umbrales_personalizados = load_config()

    # Vectorize corpus
    vectorizer = TfidfVectorizer()
    corpus = df["text"].tolist()  # List of all texts
    vectorizer.fit(corpus)

    # Apply categorization to each article
    df["label"] = df["text"].apply(
        lambda x: categorizar_articulo_subtemas_en(x, subtemas_palabras_clave, vectorizer, umbrales_personalizados, top_n=3)
    )

    # Save the DataFrame with the new labels
    df.to_csv("labelled_data.csv", index=False, encoding='utf-8')
    print("Categorización completada. Archivo guardado como labelled_data.csv")
def evalua_model(model, val_dataset, mlb):
    y_true = []
    y_pred = []
    for batch in val_dataset:
        X, y = batch
        y_true.extend(y.numpy())
        y_pred.extend(model.predict(X).numpy())
    
    y_true = mlb.inverse_transform(y_true)
    y_pred = mlb.inverse_transform(y_pred)
    
    print(classification_report(y_true, y_pred))

#example of main.py
def main():
    try:
        df = pd.read_csv("labelled_data2.csv", encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv("other_data.csv", encoding='latin-1')  # encoding alternativo
    except FileNotFoundError:
        print("File 'labelled_data2.csv' not found.")
        exit(1)


    # palabras clave para categorizacion
    # fueron cargadas a un config.json y se leen dinamicamente aca

        df = preprocess_data(df)
        
        labels_df, mlb = binarize(df)
        
        x_train, x_test, y_train, y_test = split_data(df, labels_df)
        
        bert_model, tokenizer = create_bert()
        
        train_encodings = tokenize_data_in_batches(tokenizer, x_train)
        
        val_encodings = tokenize_data_in_batches(tokenizer, x_test)

        train_dataset, val_dataset = create_tf_datasets(train_encodings, y_train, val_encodings, y_test)
        
        tuned_bert = tune_bert(train_dataset, val_dataset,num_labels=labels_df.shape[1])
        # Si necesitamos el fine tuning del bert podemos pasarlo como parametro aca
        cnn_model = create_cnn(bert_model, num_labels=labels_df.shape[1])
        
        history = train_mixed_model(cnn_model, train_dataset, val_dataset)
        
        evalua_model(cnn_model,val_dataset,mlb)

        cnn_model.save('bert_cnn_cybersecurity_model.h5')
        joblib.dump(mlb, 'mlb.pkl')

    

if __name__ == '__main__':
    main()
