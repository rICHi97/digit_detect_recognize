# 按钮
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
#* from PyQt5.QtGui import QIcon
class Ui_mainWindow(object):
    def Ui(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.setWindowModality(QtCore.Qt.WindowModal)
        mainWindow.resize(400,500)

        self.centralWidget = QtWidgets.QWidget(mainWindow)
        self.centralWidget.setObjectName("centralWidget")
        
        font = QtGui.QFont()
        font.setFamily('微软雅黑')
        font.setBold(True)
        font.setPointSize(13)
        font.setWeight(55)
        
        self.pushButton1 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton1.setGeometry(QtCore.QRect(10, 10, 120, 30))
        self.pushButton1.setObjectName("button1")
        self.pushButton1.setText("选择图片")
        self.pushButton1.setFont(font)
        
        self.pushButton2 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton2.setGeometry(QtCore.QRect(10, 70, 120, 30))
        self.pushButton2.setObjectName("button1")
        self.pushButton2.setText("自动识别")
        self.pushButton2.setFont(font)

        mainWindow.setCentralWidget(self.centralWidget)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)
    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "端子自动识别"))
        #* mainWindow.setWindowIcon(QIcon('图片.png'))
        #* 在主界面标题前面插入图片，需要图片和程序在同一路径。
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = Ui_mainWindow()
    ui.Ui(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())