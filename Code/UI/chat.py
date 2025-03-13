import sys
import os
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QScrollArea, QSizePolicy
from PySide6.QtCore import Qt
import tensorflow as tf
from transformers import BertTokenizer
ruta_carpeta = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'modulos'))
sys.path.append(ruta_carpeta)
class ChatBubble(QLabel):
     def __init__(self, message, is_sent_by_user):
        super().__init__()
        self.setText(message)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {'#a8e6cf' if is_sent_by_user else '#dcedc1'};
                border-radius: 10px;
                padding: 10px;
                word-wrap: break-word;
            }}
        """)

class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.cargar_modelo()
        self.initUI()


    def initUI(self):
        self.setWindowTitle('Ventana de Chat')
        self.setGeometry(300, 600, 300, 400)
        self.setMinimumSize(300, 400)  # Ancho y alto mínimos
        self.setMaximumSize(800, 600)

        self.layout = QVBoxLayout()
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout()
        self.chat_widget.setLayout(self.chat_layout)
        
        self.scroll_area.setWidget(self.chat_widget)
        
        self.layout.addWidget(self.scroll_area)

        self.input_layout = QHBoxLayout()
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText('Escribe tu mensaje...')
        self.send_button = QPushButton('Enviar')
        self.send_button.clicked.connect(self.send_message)
        self.input_layout.addWidget(self.message_input)
        self.input_layout.addWidget(self.send_button)
        
        self.layout.addLayout(self.input_layout)
        self.setLayout(self.layout)

    def send_message(self):
        message = self.message_input.text()
        if message:
            self.add_message(message, is_sent_by_user=True)
            self.message_input.clear()
            self.predecir(message)

    def add_message(self, message, is_sent_by_user):
        bubble = ChatBubble(message, is_sent_by_user)
        self.chat_layout.addWidget(bubble, alignment=Qt.AlignRight if is_sent_by_user else Qt.AlignLeft)
        

        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

    def receive_message(self, message):
        self.add_message(message, is_sent_by_user=False)

    def cargar_modelo(self):
        ruta_modelo = "model\model"
        self.modelo = tf.saved_model.load(ruta_modelo)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        return 
    def predecir(self, texto):
        tokens = self.tokenizer(texto, return_tensors='tf')
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        token_type_ids = tokens.get('token_type_ids')

        funcion_prediccion = self.modelo.signatures['serving_default']
        prediccion = funcion_prediccion(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = prediccion['logits']
        probabilidades = tf.nn.softmax(logits, axis=-1)
        
        # Obtener el índice de la categoría con mayor probabilidad
        categoria_mayor_probabilidad = tf.argmax(probabilidades, axis=-1).numpy()
        
        # Asumiendo que tienes una lista de nombres de categorías
        nombres_categorias = ["Categoría 1", "Categoría 2", "Categoría 3"]  # Ajusta esta lista según tu modelo
        categoria = nombres_categorias[categoria_mayor_probabilidad[0]]
        print (probabilidades)
        self.receive_message(f'La categoría con mayor probabilidad es: {categoria} ({probabilidades[0][categoria_mayor_probabilidad[0]]:.2f}%)')
        self.receive_message(f'Predicción: {prediccion}')



