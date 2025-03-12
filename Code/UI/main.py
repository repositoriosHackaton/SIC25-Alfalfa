import sys
#from ia_handler import IAHandler
from PySide6.QtWidgets import (QApplication, QMainWindow,
QPushButton, QLabel, QVBoxLayout, QWidget, QLineEdit)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice
import UI.chat

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Athena - Panel principal")
        self.setGeometry (100, 100, 400, 300)

        #No estoy segura si esto funciona----------------------------


        self.cargar_boton = QPushButton('Cargar Modelo', self)
        self.cargar_boton.clicked.connect(self.cargar_modelo)

        self.datos_input = QLineEdit(self)

        self.predecir_boton = QPushButton('Hacer Predicción', self)
        self.predecir_boton.clicked.connect(self.hacer_prediccion)

        self.resultado_label = QLabel('Resultado:', self)
    

     #-------------------------------------------------------------

        loader = QUiLoader()
        file = QFile("Code/UI/ui/main.ui")

        if not file.open(QIODevice.ReadOnly):
            print(f"No se puede abrir el archivo: {file.errorString()}")
            exit(-1)
        
        self.ui = loader.load(file, self)
        file.close()

        if not self.ui:
            print(loader.errorString())
            exit(-1) 

        layout = QVBoxLayout()
        layout.addWidget(self.ui)
        layout.addWidget(self.cargar_boton)
        layout.addWidget(self.datos_input)
        layout.addWidget(self.predecir_boton)
        layout.addWidget(self.resultado_label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.setup_connections()



#Conexiónes
    def setup_connections(self):
        btn_chat = self.ui.findChild(QPushButton, 'btn_chat')
        btn_chat.clicked.connect(self.chat_window)
        btn_est = self.ui.findChild(QPushButton, 'btn_est')
        btn_est.clicked.connect(self.est_window)
        pass

    def chat_window(self):
       
        #self.hide()
        pass 
    def est_window(self):
        print("¡Botón de estadísticas presionado!")



