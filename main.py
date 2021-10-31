from PyQt5 import QtCore, QtGui, QtWidgets
import torch
from transformers import AutoTokenizer, AutoConfig
from modeling import BertForSentimentClassification
from analyze import classify_sentiment
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1297, 748)
        self.model_name_or_path = './models/128_all_model/'

        #Configuration for the desired transformer model
        self.config = AutoConfig.from_pretrained(self.model_name_or_path)
        #Create the model with the desired transformer model
        self.model = BertForSentimentClassification.from_pretrained(self.model_name_or_path,config=self.config)

        self.model.eval()

        #Initialize the tokenizer for the desired transformer model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(0, 0, 511, 731))
        self.widget.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-image: url(./keli.png);")
        self.widget.setObjectName("widget")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(520, 60, 121, 31))
        self.label_2.setStyleSheet("font: 75 20pt \"楷体\";")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(510, 250, 151, 41))
        self.label_3.setStyleSheet("font: 75 20pt \"楷体\";")
        self.label_3.setObjectName("label_3")
        self.outputEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.outputEdit.setGeometry(QtCore.QRect(660, 240, 631, 51))
        self.outputEdit.setStyleSheet("font: 16pt \"楷体\";")
        self.outputEdit.setObjectName("outputEdit")
        self.SendButton = QtWidgets.QPushButton(self.centralwidget)
        self.SendButton.setGeometry(QtCore.QRect(840, 130, 231, 81))
        self.SendButton.setStyleSheet("QPushButton{\n"
"border-style:outset;\n"
"font: 16pt \"楷体\";\n"
"background-color:rgb(41, 238, 255);\n"
"border-radius:10px;\n"
"}\n"
"QPushButton:pressed{\n"
"border-style:inset;\n"
"border-radius:10px;\n"
"background-color:rgb(255, 255, 255);\n"
"font: 16pt \"楷体\";\n"
"}")
        self.SendButton.setObjectName("SendButton")
        self.SendButton.clicked.connect(self.push_SendButton)
        self.inputEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.inputEdit.setGeometry(QtCore.QRect(660, 50, 631, 51))
        self.inputEdit.setStyleSheet("font: 16pt \"楷体\";")
        self.inputEdit.setText("")
        self.inputEdit.setObjectName("inputEdit")
        self.result_widget = QtWidgets.QLabel(self.centralwidget)
        self.result_widget.setGeometry(QtCore.QRect(750, 320, 394, 394))
        self.result_widget.setStyleSheet("background-color: rgb(255, 255, 255);"
"border-image: url(./begin.jpeg);")
        self.result_widget.setObjectName("result_widget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    def push_SendButton(self):
        sentence=self.inputEdit.text()
        label,result = classify_sentiment(sentence,self.model,self.tokenizer)
        self.outputEdit.setText(result)
        if label == True:
            jpg = QtGui.QPixmap("./pos.jpeg").scaled(self.result_widget.width(), self.result_widget.height())
            self.result_widget.setPixmap(jpg)
        else:
            jpg = QtGui.QPixmap("./neg.jpg").scaled(self.result_widget.width(), self.result_widget.height())
            self.result_widget.setPixmap(jpg)
        return
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "基于Bert的文本情感分析系统V1.0"))
        self.label_2.setText(_translate("MainWindow", "请输入："))
        self.label_3.setText(_translate("MainWindow", "分析结果:"))
        self.outputEdit.setToolTip(_translate("MainWindow", "<html><head/><body><p>Type something ..</p></body></html>"))
        self.outputEdit.setText(_translate("MainWindow", "请在上面的聊天框输入文本"))
        self.SendButton.setText(_translate("MainWindow", "开始分析"))
        self.inputEdit.setToolTip(_translate("MainWindow", "<html><head/><body><p>Type something ..</p></body></html>"))
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_()) 