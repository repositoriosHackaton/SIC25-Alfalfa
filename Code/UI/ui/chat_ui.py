# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'chat.ui'
##
## Created by: Qt User Interface Compiler version 6.8.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QListView, QMainWindow,
    QPushButton, QSizePolicy, QTextEdit, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 467)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.msg_box = QTextEdit(self.centralwidget)
        self.msg_box.setObjectName(u"msg_box")
        self.msg_box.setMaximumSize(QSize(16777215, 50))

        self.gridLayout.addWidget(self.msg_box, 1, 0, 1, 1)

        self.bttn_snd = QPushButton(self.centralwidget)
        self.bttn_snd.setObjectName(u"bttn_snd")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bttn_snd.sizePolicy().hasHeightForWidth())
        self.bttn_snd.setSizePolicy(sizePolicy)
        self.bttn_snd.setMaximumSize(QSize(16777215, 50))

        self.gridLayout.addWidget(self.bttn_snd, 1, 1, 1, 1)

        self.chat_fld = QListView(self.centralwidget)
        self.chat_fld.setObjectName(u"chat_fld")

        self.gridLayout.addWidget(self.chat_fld, 0, 0, 1, 2)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.bttn_snd.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
    # retranslateUi

