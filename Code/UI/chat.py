import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QScrollArea, QSizePolicy
from PySide6.QtCore import Qt
import modulos.ia_handler as ia_handler
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

    def add_message(self, message, is_sent_by_user):
        bubble = ChatBubble(message, is_sent_by_user)
        self.chat_layout.addWidget(bubble, alignment=Qt.AlignRight if is_sent_by_user else Qt.AlignLeft)
        

        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

    def receive_message(self, message):
        self.add_message(message, is_sent_by_user=False)

    def cargar_modelo(self):
        self.ia_handler.cargar_modelo()
        self.receive_message('Modelo cargado correctamente')

    def hacer_prediccion(self):
        datos_entrada = self.datos_input.text()
        prediccion = ia_handler.predecir(datos_entrada)
        self.receive_message(f'Predicción: {prediccion}')
window = ChatWindow()
window.show()

