
import time
import os
import pandas as pd
import numpy as np
from math import asin, sqrt, sin, cos, tan, floor, acos
from PyQt6 import uic
from PyQt6 import QtCore
from PyQt6 import QtGui
import OpenGL.GLUT as glut
import OpenGL.GLU as glu
import OpenGL.GL as gl
from OpenGL import GLU
from OpenGL.arrays import vbo
from PyQt6 import QtOpenGL
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtOpenGLWidgets import QOpenGLWidget as QOGLW
from PyQt6.QtWidgets import QApplication, QMainWindow,QTableWidget,QTableWidgetItem,QTabWidget,QLineEdit, QDialogButtonBox, QMessageBox, QComboBox, QWidgetAction, QMenuBar,QMenu
from PyQt6.QtWidgets import QFileDialog, QDialogButtonBox, QDialog, QVBoxLayout, QLabel
from pathlib import Path
from PyQt5.QtGui import QPixmap
from SupportCodes import getkb, CharLine
import sys
Form, Window = uic.loadUiType("DWC_Form1.ui")
PlyForm, PlyWindow = uic.loadUiType("DWC_PlyWindow.ui")
ReinfForm, ReinfWindow = uic.loadUiType("DWC_ReinfWindow.ui")
MatrixForm, MatrixWindow = uic.loadUiType("DWC_MatrixWindow.ui")
app = QApplication([])
window = Window()
plyWindow = PlyWindow()
plyWindow.setModal(True)
reinfWindow = ReinfWindow()
reinfWindow.setModal(True)
matrixWindow = MatrixWindow()
matrixWindow.setModal(True)
form = Form()
form.setupUi(window)
plyform = PlyForm()
plyform.setupUi(plyWindow)
reinfform = ReinfForm()
reinfform.setupUi(reinfWindow)
matrixform = MatrixForm()
matrixform.setupUi(matrixWindow)
window.show()

plyform.tabW.setTabEnabled(1, True)
plyform.tabW.setTabEnabled(2, False)
plyform.tabW.setTabEnabled(3, False)
plyform.tabW.setTabEnabled(4, False)

#pip = QPixmap()
#pip.load("./DREAM-COMP.jpg")

isPrjNew = True

prjName = "/NewDWCPrj.txt"
prjPath = os.getcwd() + prjName
prjList = []


partTypes = ['Цилиндр', 'бак с сфер. днищами']
currentTypeIndex = 0

strategys = ['Спираль','Винт','Полярная']
stratIndex = [0]
plys = [] # массив слоев
#[stratIndex, betta, gamma, plyXmin, plyXmax, hl, FirstX, Lreif, Mmatrix, KIM, time]
reinfBase = [[]] # = [['Уголь Широкий', 5, 0.2, 0.5], ['Стекло Узкое', 5, 0.2, 0.5]]
matrixBase = [[]] # = [['Neopox 347 + 45М', 2, 0.5, 60], ['ЭД-20 + ПЕПА', 2, 0.1, 60]]
Speed = [2500, 2500, 10000, 5000] #Скорости движения X, Y, Z, A
currentReinf = reinfBase[0]
currentMatrix = matrixBase[0]
currentPly = [0]
bGcode = []
BufPlySet = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
mainGcodes = []
supplyGcodes = []
Gcode = []
packiscalc = [False]
plyiscalc = False
class config: # Определение конфигурации станка
    def __init__(self):
        self.Name = 'DREAM WINDER KEENETIC'
        self.Rring = 0

        self.DMaxLim = 300
        self.DMinLim = 5

        self.XMaxLim = 3000
        self.XMinLim = -550
        self.Xlim = 3000

        self.bettaMaxLim = 80
        self.bettaMinLim = 1

        self.hlMaxLim = 10
        self.hlMinLim = 1

        self.hMaxLim = 40
        self.hMinLim = 1

        self.thickMaxLim = 3
        self.thickMinLim = 0.05

        self.ReinfParamMaxLim = 1
        self.ReinfParamMinLim = 0

        self.GMaxLim = 20
        self.GMinLim = 0.5

        self.ratioMaxLim = 1
        self.ratioMinLim = 0.01

        self.tempMaxLim = 300
        self.tempMinLim = 20

        self.moveTime = 0.1
        self.startTime = 0.17
        self.finishTime = 1
        self.X = 0
        self.Y = 0
        self.Z = 0
        self.A = 0
        self.Lreinf = 0
        self.LreinfPos = 0
        self.D = 0
        self.betta = 90

    def getMoveOnCom(self, dX=0.0, dY=0.0, dZ=0.0, betta=None, F=0.0, PosFlag = True, Coment = None, DoFlag = True):
        if betta != None:
            self.betta = betta
            if betta >= 0:
                self.A = (90 - betta)
            else:
                self.A = -(90 + betta)
        self.X += dX
        if DoFlag:
            self.Y += dY
        self.Z += dZ
        if DoFlag:
            comand = 'G1 X{0:.4f} Y{1:.4f} Z{2:.4f} A{3:.4f} F{4:.0f}'.format(self.X, self.Y, self.Z, self.A, F)
            if Coment != None:
                comand += '({0})'.format(Coment)
            comand += '\n'
            self.Lreinf += getLreinf(self.D, dZ, self.betta)
            if PosFlag:
                self.LreinfPos += getLreinf(self.D, dZ, self.betta)
            return comand


Config = config()

def sendMess(text):
    msg = QMessageBox()
    msg.setWindowTitle("Предупреждение")
    msg.setText(text)
    msg.exec()
def checkInputReinfForm():
    try:
        h = float(reinfform.hle.text())  # ширина ленты
    except:
        reinfform.hle.setText(str('{:.1f}'.format(Config.hMaxLim)))
        h = Config.hMinLim
        sendMess('некорректная толщина ленты')
    if h > Config.hMaxLim:
        reinfform.hle.setText(str('{:.1f}'.format(Config.hMaxLim)))
    if h < Config.hMinLim:
        reinfform.hle.setText(str('{:.1f}'.format(Config.hMinLim)))

    try:
        thick = float(reinfform.thickle.text())  # толщина слоя
    except:
        reinfform.thickle.setText(str('{:.1f}'.format(Config.thickMinLim)))
        thick = Config.thickMinLim
        sendMess('некорректная толщина слоя')

    if thick > Config.thickMaxLim:
        reinfform.thickle.setText(str('{:.1f}'.format(Config.thickMaxLim)))
    if thick < Config.thickMinLim:
        reinfform.thickle.setText(str('{:.1f}'.format(Config.thickMinLim)))

    try:
        Param = float(reinfform.Paramle.text())  # параметр
    except:
        reinfform.Paramle.setText(str('{:.1f}'.format(Config.ReinfParamMaxLim)))
        Param = Config.ReinfParamMaxLim
        sendMess('некорректный параметр')
    if Param > Config.ReinfParamMaxLim:
        reinfform.Paramle.setText(str('{:.1f}'.format(Config.ReinfParamMaxLim)))
    if Param < Config.ReinfParamMinLim:
        reinfform.Paramle.setText(str('{:.1f}'.format(Config.ReinfParamMinLim)))
def checkInputMatrixForm():
    try:
        G = float(matrixform.Gle.text())  # Расход
    except:
        matrixform.Gle.setText(str('{:.1f}'.format(Config.GMaxLim)))
        G = Config.GMaxLim
        sendMess('некорректный расход')
    if G > Config.GMaxLim:
        matrixform.Gle.setText(str('{:.1f}'.format(Config.GMaxLim)))
    if G < Config.GMinLim:
        matrixform.hle.setText(str('{:.1f}'.format(Config.GMinLim)))

    try:
        ratio = float(matrixform.ratiole.text())  # Соотношение отвердитель/смола
    except:
        matrixform.ratiole.setText(str('{:.1f}'.format(Config.ratioMaxLim)))
        ratio = Config.ratioMaxLim
        sendMess('некорректное соотношение')

    if ratio > Config.ratioMaxLim:
        matrixform.ratiole.setText(str('{:.1f}'.format(Config.ratioMaxLim)))
    if ratio < Config.ratioMinLim:
        matrixform.ratiole.setText(str('{:.1f}'.format(Config.ratioMinLim)))

    try:
        temp = float(matrixform.temple.text())  # Температура
    except:
        matrixform.temple.setText(str('{:.1f}'.format(Config.tempMaxLim)))
        temp = Config.tempMaxLim
        sendMess('некорректная температура')

    if temp > Config.tempMaxLim:
        matrixform.temple.setText(str('{:.1f}'.format(Config.tempMaxLim)))
    if temp < Config.tempMinLim:
        matrixform.temple.setText(str('{:.1f}'.format(Config.tempMinLim)))
def checkGcodebtn():
    if packiscalc[0] == True:
        form.Gcodebtn.setEnabled(True)
    else:
        form.Gcodebtn.setEnabled(False)
def upReinfTable():
    reinfform.reinftbl.clearContents()
    reinfform.reinftbl.setRowCount(len(reinfBase))
    for i in range(0, len(reinfBase)):
        reinfform.reinftbl.setItem(i, 0, QTableWidgetItem(reinfBase[i][0]))
        for j in range(1, len(reinfBase[i])):
            reinfform.reinftbl.setItem(i, j, QTableWidgetItem(str(str('{:.1f}'.format(reinfBase[i][j])))))
def upMatrixTable():
    matrixform.matrixtbl.clearContents()
    matrixform.matrixtbl.setRowCount(len(matrixBase))
    for i in range(0, len(matrixBase)):
        matrixform.matrixtbl.setItem(i, 0, QTableWidgetItem(matrixBase[i][0]))
        for j in range(1, len(matrixBase[i])):
            matrixform.matrixtbl.setItem(i, j, QTableWidgetItem(str(str('{:.1f}'.format(matrixBase[i][j])))))
def upComboBoxes():
    form.reinfcb.clear()
    global reinfBase
    for reinf in reinfBase:
        form.reinfcb.addItem(str(reinfBase.index(reinf)+1) + '. ' + reinf[0])

    global matrixBase
    form.matrixcb.clear()
    for matrix in matrixBase:
        form.matrixcb.addItem(str(matrixBase.index(matrix) + 1) + '. ' + matrix[0])
def upTypeComboBox():
    for item in partTypes:
        form.typecb.addItem(str(partTypes.index(item) + 1) + '. ' + item)
upTypeComboBox()
def writeMaterialxlsx():
    writer = pd.ExcelWriter('MaterialBase.xlsx', engine='xlsxwriter')
    df = pd.DataFrame(data=reinfBase, columns=['Название','Ширина ленты', 'толщина', '...'])
    df.to_excel(writer, 'Наполнители')
    dm = pd.DataFrame(data=matrixBase, columns=['Название', 'расход', 'соотношение отвердитель/смола', 'температура полимеризации'])
    dm.to_excel(writer, 'Связующие')
    writer._save()
def readMaterialxlsx():
    file = 'MaterialBase.xlsx'
    xl = pd.ExcelFile(file)
    df = xl.parse('Наполнители')
    dm = xl.parse('Связующие')
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    dm.drop(columns=dm.columns[0], axis=1, inplace=True)
    global reinfBase
    global matrixBase
    reinfBase = df.values.tolist()
    matrixBase = dm.values.tolist()
readMaterialxlsx()
upComboBoxes()
def setReinfLe():
    global reinfBase
    reifindex = reinfform.reinftbl.currentRow()
    reinfform.Namele.setText(reinfBase[reifindex][0])
    reinfform.hle.setText(str(reinfBase[reifindex][1]))
    reinfform.thickle.setText(str(reinfBase[reifindex][2]))
    reinfform.Paramle.setText(str(reinfBase[reifindex][3]))
def setMatrixLe():
    global matrixBase
    matrixindex = matrixform.matrixtbl.currentRow()
    matrixform.Namele.setText(matrixBase[matrixindex][0])
    matrixform.Gle.setText(str(matrixBase[matrixindex][1]))
    matrixform.ratiole.setText(str(matrixBase[matrixindex][2]))
    matrixform.temple.setText(str(matrixBase[matrixindex][3]))
def setMatrixWindow():
    matrixWindow.show()
    upMatrixTable()
    upCurrentMatrix()
    matrixindex = form.matrixcb.currentIndex()
    matrixform.matrixtbl.selectRow(matrixindex)
    setMatrixLe()
def setReinfWindow():
    reinfWindow.show()
    upReinfTable()
    upCurrentReinf()
    reinfindex = form.reinfcb.currentIndex()
    reinfform.reinftbl.selectRow(reinfindex)
    setReinfLe()
def canselReinfSet():
    reinfWindow.hide()
    upComboBoxes()
def canselMatrixSet():
    matrixWindow.hide()
    upComboBoxes()
def okReinfSet():
    global reinfBase
    reifindex = reinfform.reinftbl.currentRow()
    reinfBase[reifindex][0] = reinfform.Namele.text()
    reinfBase[reifindex][1] = float(reinfform.hle.text())
    reinfBase[reifindex][2] = float(reinfform.thickle.text())
    reinfBase[reifindex][3] = float(reinfform.Paramle.text())
    writeMaterialxlsx()
    reinfWindow.hide()
    upComboBoxes()
def okMatrixSet():
    global matrixBase
    matrixindex = matrixform.matrixtbl.currentRow()
    matrixBase[matrixindex][0] = matrixform.Namele.text()
    matrixBase[matrixindex][1] = float(matrixform.Gle.text())
    matrixBase[matrixindex][2] = float(matrixform.ratiole.text())
    matrixBase[matrixindex][3] = float(matrixform.temple.text())
    writeMaterialxlsx()
    matrixWindow.hide()
    upComboBoxes()
def addReinfSet():
    global reinfBase
    newReinf = [0, 0, 0, 0]
    newReinf[0] = reinfform.Namele.text()
    for reinf in reinfBase:
        if reinf[0] == newReinf[0]:
            reinfform.Namele.setStyleSheet("color: red;")
            return
        elif newReinf[0] == '':
            reinfform.Namele.setStyleSheet("background-color: red;")
    reinfform.Namele.setStyleSheet("color: black;")
    reifindex = reinfform.reinftbl.currentRow()
    if reifindex < 0:
        reifindex = 0
    newReinf[1] = float(reinfform.hle.text())
    newReinf[2] = float(reinfform.thickle.text())
    newReinf[3] = float(reinfform.Paramle.text())
    reinfBase.insert(reifindex,newReinf)
    writeMaterialxlsx()
    upReinfTable()
    upComboBoxes()
def addMatrixSet():
    global matrixBase
    newMatrix = [0,0,0,0]
    newMatrix[0] = matrixform.Namele.text()
    for matrix in matrixBase:
        if matrix[0] == newMatrix[0]:
            matrixform.Namele.setStyleSheet("color: red;")
            return
        elif newMatrix[0] == '':
            matrixform.Namele.setStyleSheet("background-color: red;")
    matrixform.Namele.setStyleSheet("color: black;")
    matrixindex = matrixform.matrixtbl.currentRow()
    if matrixindex < 0:
        matrixindex = 0
    newMatrix[1] = float(matrixform.Gle.text())
    newMatrix[2] = float(matrixform.ratiole.text())
    newMatrix[3] = float(matrixform.temple.text())
    matrixBase.insert(matrixindex, newMatrix)
    writeMaterialxlsx()
    upMatrixTable()
    upComboBoxes()

def delReinfSet():
    if len(reinfBase) > 0:
        reifindex = reinfform.reinftbl.currentRow()
        if reifindex < 0:
            reifindex = 0
        reinfBase.pop(reifindex)
        writeMaterialxlsx()
        upReinfTable()
        setReinfLe()
        upComboBoxes()

def delMatrixSet():
    if len(matrixBase) > 0:
        matrixindex = matrixform.matrixtbl.currentRow()
        if matrixindex < 0:
            matrixindex = 0
        matrixBase.pop(matrixindex)
        writeMaterialxlsx()
        upMatrixTable()
        setMatrixLe()
        upComboBoxes()
def upCurrentReinf():
    global currentReinf
    currentReinf = reinfBase[form.reinfcb.currentIndex()]
    packiscalc[0] = False
    checkGcodebtn()
upCurrentReinf()
def upCurrentMatrix():
    global currentMatrix
    currentMatrix = matrixBase[form.matrixcb.currentIndex()]
    packiscalc[0] = False
    checkGcodebtn()
upCurrentMatrix()

def upCurrentType():
    global currentTypeIndex
    currentTypeIndex = form.typecb.currentIndex()
    packiscalc[0] = False
    checkGcodebtn()
def upplytable():
    form.plytbl.clearContents()
    form.plytbl.setRowCount(len(plys))
    for i in range(0, len(plys)):
        form.plytbl.setItem(i, 0, QTableWidgetItem(strategys[plys[i][0]]))
        for j in range(1, len(plys[i])):
            form.plytbl.setItem(i, j, QTableWidgetItem(str(str('{:.1f}'.format(plys[i][j])))))
upplytable()

def showWindow():
    plyWindow.show()
def setPlyWindow(n):
    currentPly[0] = n
    if plys[currentPly[0]] == []:
        stratIndex[0] = 0
    else:
        stratIndex[0] = plys[n][0]
    plyform.tabW.setCurrentIndex(stratIndex[0])

def plyCalcbtn():
    stratindex = plyform.tabW.currentIndex()
    plyCalc(stratindex)
def plyCalc(stratindex):
    if stratindex == 0:
        plyCalcSpirale()
    elif stratindex == 1:
        plyCalcVint()

def getLreinf(D, Z, betta):
    if betta == 0:
        betta = 10
    return (3.1415 * D * Z / 360) / sin(betta * 3.1415 / 180)


def getTimefromGcode(Gcode = [''], k = 1.33):
    time = 0
    for line in Gcode:
        if line != None:
            if line.find('G1') != -1:
                X = 0
                Y = 0
                A = 0
                Z = 0
                F = 8000
                i = line.find('X')
                if i != -1:
                    X = float(line[i+1:i+4])
                i = line.find('Y')
                if i != -1:
                    Y = float(line[i + 1:i + 4])
                i = line.find('Z')
                if i != -1:
                    Z = float(line[i + 1:i + 4])
                i = line.find('A')
                if i != -1:
                    A = float(line[i + 1:i + 4])
                i = line.find('F')
                if i != -1:
                    F = float(line[i + 1:line.find('\n')])
                time = time + ((X**2 + Y**2 + Z**2 + A**2)**0.5)/F
        return time * k
def plyCalcVint():
    global  mainGcodes
    global bGcode
    #блок расчета траектории "Винт"
    stratIndex[0] = 1
    D = float(form.Dle.text())  # Диаметр
    hl = float(plyform.hlle2.text())  # сменение в лентах
    plyXmin = float(plyform.plyXminle2.text())  # координата левого края
    plyXmax = float(plyform.plyXmaxle2.text())  # координата правого края
    L = plyXmax - plyXmin  # длина слоя
    h = currentReinf[1]  # ширина ленты
    betta = acos(h*hl / (3.1415*D))# угол намотки

    invert_x_flag = 1

    returncheck = plyform.ReturnCB2.isChecked()
    print(returncheck)

    dX = h * hl / sin(betta)
    plyform.dxlb2.setText('Смещение dX: ' + str('{:.1f}'.format(dX)) + ' мм.')

    prRot = L / dX + 2
    prRotDeg = prRot * 360
    plyform.prRotlb2.setText('Оборотов на 1 пр.: ' + str('{:.1f}'.format(prRot)) + ' шт.')

    betta = betta*180/3.1415
    plyform.bettalb2.setText('Угол намотки: ' + str('{:.1f}'.format(betta)) + ' град.')

    # расчет массы связующего и длины ленты, запись буферного G-coda
    bufferG = []
    bufferG.clear()
    Lreinf = 1
    LreinfPos = 0  # полезная длина ленты наполнителя
    Mmatrix = 0
    cycles = 1
    Z = 0
    X = 0
    Y = 10
    time = 0  # Время, мин
    # расчет вылета
    R = D / 2
    L1 = ((R + Y) ** 2 + (Config.Rring * sin(betta * 3.1415 / 180)) ** 2 - R ** 2) ** 0.5
    X1 = 1 / tan(betta * 3.1415 / 180) * L1
    Xout = X1 + Config.Rring * cos(betta * 3.1415 / 180)
    Xout1 = X1 + 2 * Config.Rring * cos(betta * 3.1415 / 180) + dX
    plyform.Xoutlb2.setText('Вылет отрезки: ' + '{:.1f}'.format(Xout1) + ' мм')
    Zstatic = 360
    Zrun = 20
    Ka = 0.3  # часть оборота без смещения
    A = -(90 - betta)
    X = plyXmin - Xout - dX
    bufferG.append('G1 A' + str('{:.4f}'.format(A*invert_x_flag)) + ' F' + str('{:.0f}'.format(Speed[3])) + '\n')
    bufferG.append('G1 X' + str('{:.4f}'.format(X*invert_x_flag)) + ' F' + str('{:.0f}'.format(Speed[0])) + '\n')

    Z = Z + (Zstatic - Zrun) * (1 - Ka)
    A = +(90 - betta)
    bufferG.append('G1 A' + str('{:.4f}'.format(A*invert_x_flag)) + ' Z' + str('{:.4f}'.format(Z)) + ' F' + str(
        '{:.0f}'.format(Speed[2])) + '\n')

    Z = Z + (Zstatic - Zrun) * Ka
    bufferG.append('G1 Z' + str('{:.4f}'.format(Z)) + ' F' + str('{:.0f}'.format(Speed[2])) + '\n')

    Z = Z + Zrun
    X = X + 2 * Xout
    bufferG.append('G1 X' + str('{:.4f}'.format(X*invert_x_flag)) + ' Z' + str('{:.4f}'.format(Z)) + ' F' + str(
        '{:.0f}'.format(Speed[2])) + '\n')

    procent = 0.05 * 100
    bufferG.append('(Слой № ' + str(currentPly[0] + 1) + ' ' + str('{:.1f}'.format(procent)) + '% ) \n')

    Z = Z + prRotDeg
    X = X + L + 2 * dX
    bufferG.append('G1 X' + str('{:.4f}'.format(X*invert_x_flag)) + ' Z' + str('{:.4f}'.format(Z)) + ' F' + str(
        '{:.0f}'.format(Speed[2])) + '\n')

    Z = Z + (Zstatic - Zrun) * (1 - Ka)
    A = (90 - betta)
    bufferG.append('G1 A' + str('{:.4f}'.format(A*invert_x_flag)) + ' Z' + str('{:.4f}'.format(Z)) + ' F' + str(
        '{:.0f}'.format(Speed[2])) + '\n')

    Z = Z + (Zstatic - Zrun) * Ka
    bufferG.append('G1 Z' + str('{:.4f}'.format(Z)) + ' F' + str('{:.0f}'.format(Speed[2])) + '\n')
    if returncheck:
        cycles = cycles + 1
        procent = 50
        bufferG.append('(Слой № ' + str(currentPly[0] + 1) + ' ' + str('{:.1f}'.format(procent)) + '% ) \n')

        Z = Z + Zrun
        X = X - 2 * Xout
        bufferG.append('G1 X' + str('{:.4f}'.format(X*invert_x_flag)) + ' Z' + str('{:.4f}'.format(Z)) + ' F' + str(
            '{:.0f}'.format(Speed[2])) + '\n')

        Z = Z + prRotDeg
        X = X - L - 2 * dX
        bufferG.append('G1 X' + str('{:.4f}'.format(X*invert_x_flag)) + ' Z' + str('{:.4f}'.format(Z)) + ' F' + str(
            '{:.0f}'.format(Speed[2])) + '\n')

        Z = Z + (Zstatic) * (1 - Ka)
        A = (90 - betta)
        bufferG.append('G1 A' + str('{:.4f}'.format(A*invert_x_flag)) + ' Z' + str('{:.4f}'.format(Z)) + ' F' + str(
            '{:.0f}'.format(Speed[2])) + '\n')

        Z = Z + (Zstatic) * Ka
        bufferG.append('G1 Z' + str('{:.4f}'.format(Z)) + ' F' + str('{:.0f}'.format(Speed[2])) + '\n')

    A = 0
    bufferG.append('G1 A' + str('{:.4f}'.format(A)))

    procent = 100
    bufferG.append('(Слой № ' + str(currentPly[0] + 1) + ' ' + str('{:.1f}'.format(procent)) + '% ) \n')

    #Lreinf = Lreinf + (3.1415 * D * prRot) / sin(betta * 3.1415 / 180)  # Проход вправо
    #LreinfPos = LreinfPos + (3.1415 * D * prRot) / sin(betta * 3.1415 / 180) * (L / (L + 2 * Xout))
    #Lreinf = Lreinf + 3.1415 * D * (addangle1 + Zstatic) / 360  # Проворот
    #Lreinf = Lreinf + (3.1415 * D * prRot) / sin(betta * 3.1415 / 180)  # Проход влево
    #LreinfPos = LreinfPos + (3.1415 * D * prRot) / sin(betta * 3.1415 / 180) * (L / (L + 2 * Xout))
    #Lreinf = Lreinf + 3.1415 * D * (addangle2 + Zstatic - gamma) / 360  # Проворот

    #time = time + 4 * (90 - betta) / Speed[3]  # повороты осей
    #time = time + 2 * (prRotDeg ** 2 + (L + 2 * Xout) ** 2) ** 0.5 / Speed[2]  # проходы вправо и влево
    #time = time + ((270 + addangle1) ** 2 + 10 ** 2) ** 0.5 / Speed[2]  # проворот 1
    #time = time + ((270 + addangle2 - gamma) ** 2 + 10 ** 2) ** 0.5 / Speed[2]  # проворот 2
    #time = time * 1.33

    Lreinf = Lreinf / 1000  # Длина ленты наполнителя
    LreinfPos = LreinfPos / 1000
    plyform.Lreinflb2.setText('Длина ленты: ' + str('{:.1f}'.format(Lreinf)) + ' м')

    Mmatrix = currentMatrix[1] * Lreinf  # Масса связующего
    plyform.Mmatrixlb2.setText('Масса связующего: ' + str('{:.1f}'.format(Mmatrix)) + ' г')

    Mharder = Mmatrix * currentMatrix[2] / (currentMatrix[2] + 1)  # Масса связующего
    plyform.Mharderlb2.setText('Масса отвердителя: ' + str('{:.1f}'.format(Mharder)) + ' г')

    plyform.cycleslb2.setText('Циклов: ' + str('{:.1f}'.format(cycles)) + ' шт.')

    KIM = LreinfPos / Lreinf  # Коэф использования материала (по наполнителю)
    plyform.KIMlb2.setText('КИМ: ' + str('{:.1f}'.format(KIM * 100)) + ' %')

    gamma = (360 * h) / (3.1415 * D * cos(betta * 3.1415 / 180))
    plyform.gammalb2.setText('Угол смещения Gamma: ' + str('{:.1f}'.format(gamma)) + ' град.')

    kspiral = 3  # Утолшение для стратегии спираль
    thick = currentReinf[2] * 2 * kspiral
    plyform.thicklb2.setText('Толщина: ' + str('{:.1f}'.format(thick)) + ' мм')

    # Время выполнения
    plyform.timelb2.setText('Время: ' + str('{:.1f}'.format(time)) + ' мин.')

    FirstX = plyXmin
    BufPlySet[0] = [stratIndex[0], betta, gamma, plyXmin, plyXmax, hl, FirstX, Lreinf, Mmatrix, KIM, time, cycles,
                    thick, Xout1]
    bGcode = bufferG
    global plyiscalc
    plyiscalc = True
    checkPlyCalc()

def plyCalcSpirale():
    global mainGcodes
    global bGcode
    CylFlag = False
    # блок расчета траектории "Спираль"
    D = float(form.Dle.text()) # Диаметр
    Config.D = D
    betta = float(plyform.bettale1.text()) # угол намотки
    hl = float(plyform.hlle.text()) # сменение в лентах
    plyXmin = float(plyform.plyXminle.text()) # координата левого края
    plyXmax = float(plyform.plyXmaxle.text()) # координата правого края
    RingCheck = plyform.ringChB.isChecked()
    BettaRingCheck = plyform.BettaRingChB.isChecked()

    Xring = float(plyform.Xringle.text())
    BettaRing = float(plyform.BettaRingle.text())
    L = plyXmax-plyXmin # длина слоя
    h = currentReinf[1] # ширина ленты

    cycles = int(3.1415 * D * cos(betta*3.1415/180)/h) + 1  # Количество циклов
    plyform.cyclelb.setText('Число чиклов: ' + str('{:.0f}'.format(cycles)) + ' шт')

    gamma = (360 * h)/(3.1415 * D * cos(betta*3.1415/180)) * hl # угол смещения по окружности
    plyform.gammalb.setText('угол смещения: ' + str('{:.1f}'.format(gamma)) + ' град.')

    if not RingCheck:
        Xring = 0
        plyform.Xringle.setText('{:.0f}'.format(Xring))
        plyform.Xringle.setEnabled(False)
        plyform.BettaRingle.setEnabled(False)
        plyform.BettaRingChB.setEnabled(False)
        BettaRing = acos(h * hl / (3.1415 * D)) * 180 / 3.1415
        plyform.BettaRingle.setText('{:.1f}'.format(BettaRing))
    else:
        plyform.Xringle.setText('{:.0f}'.format(Xring))
        plyform.Xringle.setEnabled(True)
        plyform.BettaRingle.setEnabled(True)
        plyform.BettaRingChB.setEnabled(True)

    if BettaRingCheck:
        BettaRing = acos(h * hl / (3.1415 * D)) * 180/3.1415
        plyform.BettaRingle.setText('{:.1f}'.format(BettaRing))

    ks = 5  # ускорение промотки на переходе betta => BettaRing
    Zstatic = 60  # промотка ленты на повороте раскладчика
    Xrun = 20  # часть траектории спираль для выхода на нормальный угол BettaRing => betta
    Zrun = ZrunFromXrun(Xrun)  # часть траектории спираль для выхода на нормальный угол BettaRing => betta
    Y = 10
    R = D / 2
    L1 = ((R + Y) ** 2 + (Config.Rring * sin(betta * 3.1415 / 180)) ** 2 - R ** 2) ** 0.5
    X1 = 1 / tan(betta * 3.1415 / 180) * L1
    Xout = 5  # + Config.Rring*cos(betta*3.1415/180)

    prRot = (L - (2* Xrun + Xring)) * tan(betta * 3.1415 / 180) / (3.1415 * D)  # Количество оборотов на проход
    plyform.prRotlb.setText('Оборотов на 1 пр. : ' + str('{:.1f}'.format(prRot)) + ' шт')

    prRotDeg = prRot * 360

    prRotRing = (Xring+Xout) * tan(BettaRing * 3.1415 / 180) / (3.1415 * D)  # Количество оборотов на проход
    #plyform.prRotlb.setText('Оборотов на 1 пр. : ' + str('{:.1f}'.format(prRot)) + ' шт')

    prRotRingDeg = prRotRing * 360

    # расчет массы связующего и длины ленты, запись буферного G-coda
    bufferG = []
    bufferG.clear()
    # полезная длина ленты наполнителя
    Mmatrix = 0
    Z = 0
    X = 0
    time = 0 # Время, мин
    #Xout = Y/tan(betta*3.1415/180) + Config.Rring*sin(betta*3.1415/180) # вылет
    Xout = X1 #+ Config.Rring*cos(betta*3.1415/180)
    Yout = 20
    #Xout1 = Xout + Config.Rring*sin(betta*3.1415/180)
    Xout1 = X1 #+ 2 * Config.Rring*cos(betta*3.1415/180)
    plyform.Xoutlb.setText('Вылет отрезки: ' + '{:.1f}'.format(Xout1) + ' мм')
    addangle1 = 0
    addangle2 = 0
    Xsmot = 0

    Config.Lreinf = 0.1
    Config.LreinfPos = 0.1
    Config.Y = 0
    Config.Z = 0
    Config.X = 0
    Config.A = 0
    bufferG.append('G92 X0 Y0 Z0\n')
    # 1 - начальная промотка слоя
    bufferG.append(Config.getMoveOnCom(dZ=Zstatic, betta=90, PosFlag=False, F=Speed[2]))

    if not RingCheck:
        BettaRing = betta
        prRotRingDeg = 0
        Zstatic = 230

    for i in range(cycles):
        # Вывод прогресса
        procent = i / cycles * 100
        bufferG.append('(Слой № ' + str(currentPly[0] + 1) + ' {:.1f} %)\n'.format(procent))

        # 2 - поворот раскладчика и отведение зонда
        bufferG.append(Config.getMoveOnCom(dY=-Yout, dZ=Zstatic, betta=BettaRing, PosFlag=False, F=Speed[2]))

        # 3 - прохождение кольцевого участка при наличии кольцевых слоев
        bufferG.append(Config.getMoveOnCom(dX=Xring, dZ=prRotRingDeg, F=Speed[2], DoFlag=RingCheck))

        # 4 поворот раскладчика на прямой угол при наличии кольцевых слоев
        bufferG.append(Config.getMoveOnCom(dX=-2, betta=50, F=Speed[2], DoFlag=RingCheck & CylFlag))

        # 5 - переход на спиральный участок (проход) при наличии кольцевых слоев
        bufferG.append(Config.getMoveOnCom(dX=Xrun+Xsmot+2, betta=0, F=Speed[2], DoFlag=RingCheck & CylFlag))

        # 6 - переход на спиральный участок (домотка) при наличии кольцевых слоев
        bufferG.append(Config.getMoveOnCom(betta=betta, F=Speed[2], DoFlag=RingCheck & CylFlag))

        # 7 - поворот раскладчика на угол намотки при наличии кольцевых слоев
        bufferG.append(Config.getMoveOnCom(dZ=Zrun * 0.5, betta=betta, F=Speed[2]))

        # 8 - намотка спирального участка до перехода
        bufferG.append(Config.getMoveOnCom(dX=(L - Xring * 1 - 2 * Xrun), dZ=prRotDeg, F=Speed[2]))

        # 9 поворот раскладчика на прямой угол при наличии кольцевых слоев
        bufferG.append(Config.getMoveOnCom(dZ=Zrun * 0.5, betta=50, F=Speed[2], DoFlag=RingCheck))

        # 10 - переход на колцевой участок (проезд) при наличии кольцевых слоев
        bufferG.append(Config.getMoveOnCom(dX=Xrun + Xsmot, dZ=Zrun * 0.5, betta=0, F=Speed[2], DoFlag=RingCheck))

        # 11 - переход на колцевой участок (домотка) при наличии кольцевых слоев
        bufferG.append(Config.getMoveOnCom(dZ=Zrun * 2.5, F=Speed[2], DoFlag=RingCheck))

        # 12 - поворот раскладчика для финальной промотки и выравнивания
        bufferG.append(Config.getMoveOnCom(dZ=Zstatic * 0.5, dX=-Xsmot, dY=+Yout, betta=45, F=Speed[2]))

        # 13 - поворот раскладчика и подвод зонда
        bufferG.append(Config.getMoveOnCom(dZ=Zstatic * 0.5, F=Speed[2], betta=90, PosFlag=False))

        # 14 - промотка слоя до целого числа оборотов
        addangle1 = ((Config.Z//360 + 1) * 360 - Config.Z)
        bufferG.append(Config.getMoveOnCom(dZ=Zstatic + addangle1 + gamma*(i+1), F=Speed[2], PosFlag=False))

        # 15 - поворот раскладчика и отвод зонда
        bufferG.append(Config.getMoveOnCom(dY=-Yout, dZ=Zstatic * 0.8, betta=-BettaRing, PosFlag=False, F=Speed[2]))

        # 16 - намотка кольцевого слоя при наличии кольцевых слоев
        bufferG.append(Config.getMoveOnCom(dX=-Xring, dZ=prRotRingDeg, F=Speed[2], DoFlag=RingCheck))

        # 17 - поворот раскладчика на прямой угол при наличии кольцевых слоев
        bufferG.append(Config.getMoveOnCom(dX=+2, betta=-50, F=Speed[2], DoFlag=RingCheck & CylFlag))

        # 18 - переход на спиральный участок (проезд) при наличии кольцевых слоев
        bufferG.append(Config.getMoveOnCom(dX=-Xrun-Xsmot-2, betta=180, F=Speed[2], DoFlag=RingCheck & CylFlag))

        # 19 - переход на спиральный участок (домотка) при наличии кольцевых слоев
        #bufferG.append(Config.getMoveOnCom(F=Speed[2], DoFlag=RingCheck & CylFlag))

        # 20 - поворот раскладчика на угол намотки
        bufferG.append(Config.getMoveOnCom(dZ=Zrun, betta=-betta, F=Speed[2]))

        # 21 - намотка спирального участка
        bufferG.append(Config.getMoveOnCom(dX=-(L - Xring * 1 - 2 * Xrun), dZ=prRotDeg, F=Speed[2]))

        # 22 поворот раскладчика на прямой угол при наличии кольцевых слоев
        bufferG.append(Config.getMoveOnCom(dZ=Zrun, betta=-50, F=Speed[2], DoFlag=RingCheck))

        # 23 - переход на колцевой участок (проезд) при наличии кольцевых слоев
        bufferG.append(Config.getMoveOnCom(dX=-Xrun - Xsmot, dZ=Zrun * 0.5, betta=180, F=Speed[2], DoFlag=RingCheck))

        # 24 - переход на кольцевой участок (домотка) при наличии кольцевых слоев
        bufferG.append(Config.getMoveOnCom(dZ=Zrun * 2.5, F=Speed[2], DoFlag=RingCheck))

        # 25 - поворот раскладчика для финальной намотки и выравнивания
        bufferG.append(Config.getMoveOnCom(dZ=Zrun * 0.5, dX=+Xsmot, dY=+Yout, betta=-45, F=Speed[2]))

        # 26 - поворот раскладчика
        bufferG.append(Config.getMoveOnCom(dZ=Zstatic, F=Speed[2], betta=90, PosFlag=False))

        # 27 - промотка слоя до целого числа оборотов
        addangle2 = ((Config.Z // 360 + 1) * 360 - Config.Z)
        bufferG.append(Config.getMoveOnCom(dZ=Zstatic + addangle2 + gamma * (i+1), F=Speed[2], PosFlag=False))
    #2.2 - поворот раскладчика

    bufferG.append('G1 A0' + ' F{:.0f}\n'.format(Speed[2] * ks))
    bufferG.append('M400\n')

    for line in bufferG:
        if line == None:
            bufferG.remove(line)

    plyform.addangle1lb.setText('Доп. угол 1: ' + str('{:.1f}'.format(addangle1)) + ' град.')

    plyform.addangle2lb.setText('Доп. угол 2: ' + str('{:.1f}'.format(addangle2)) + ' град.')
    time = getTimefromGcode(bufferG)
    Config.Lreinf = Config.Lreinf/1000 # Длина ленты наполнителя
    Config.LreinfPos = Config.LreinfPos/1000
    plyform.lreinflb.setText('Длина ленты: ' + str('{:.1f}'.format(Config.Lreinf)) + ' м')

    Mmatrix = currentMatrix[1] * Config.Lreinf # Масса связующего
    plyform.mmatrixlb.setText('Масса связующего: ' + str('{:.1f}'.format(Mmatrix)) + ' г')

    Mharder = Mmatrix * currentMatrix[2]/(currentMatrix[2] + 1) # Масса связующего
    plyform.mharderlb.setText('Масса отвердителя: ' + str('{:.1f}'.format(Mharder)) + ' г')

    KIM = Config.LreinfPos/Config.Lreinf # Коэф использования материала (по наполнителю)
    plyform.KIMlb.setText('КИМ: ' + str('{:.1f}'.format(KIM*100)) + ' %')

    kspiral = 3 # Утолшение для стратегии спираль
    thick = currentReinf[2] * 2 * kspiral
    plyform.thicklb.setText('Толщина: ' + str('{:.1f}'.format(thick)) + ' мм')

    # Время выполнения
    plyform.timelb.setText('Время: ' + str('{:.1f}'.format(time)) + ' мин.')

    FirstX = plyXmin
    BufPlySet[0] = [stratIndex[0], betta, gamma, plyXmin, plyXmax, hl, FirstX, Config.Lreinf, Mmatrix, KIM, time, cycles, thick, Xout1, Xring, Xrun]
    bGcode = bufferG
    global plyiscalc
    plyiscalc = True
    checkPlyCalc()

def ZrunFromXrun(Xrun):
    return Xrun * 1
def canselPlySet():

    if len(plys[currentPly[0]]) == 0:
        plys.pop(currentPly[0])
    plyWindow.hide()

def applyPlySet():
    mainGcodes[currentPly[0]] = bGcode
    plys[currentPly[0]] = BufPlySet[0]
    if currentPly[0] != 0: # Костыль
        plys[0] = BufPlySet[1]
    upplytable()

    packiscalc[0] = False
    checkGcodebtn()

def OkPlySet():
    global mainGcodes
    global bGcode
    mainGcodes[currentPly[0]] = bGcode
    plys[currentPly[0]] = BufPlySet[0]
    if currentPly[0] != 0: # Костыль
        plys[0] = BufPlySet[1]
    upplytable()
    plyWindow.hide()
    up2Dgraf()
    packiscalc[0] = False
    checkGcodebtn()

def addply():
    checkInputMainForm()
    #form.reinfcb.setEnabled(False)
    #form.matrixcb.setEnabled(False)
    global plys
    showWindow()
    currentPly[0] = form.plytbl.currentRow()
    if currentPly[0] < 0:
        currentPly[0] = 0
    if len(plys) == 0:
        plys = [[]]
    else:
        plys.insert(currentPly[0], [])
    BufPlySet[1] = plys[0]
    mainGcodes.insert(currentPly[0], [])
    supplyGcodes.insert(currentPly[0], [])
    setPlyWindow(currentPly[0])

    packiscalc[0] = False
    checkGcodebtn()

    global plyiscalc
    plyiscalc = False
    checkPlyCalc()

def delply():
    if len(plys)>0:
        plys.pop(form.plytbl.currentRow())
        mainGcodes.pop(form.plytbl.currentRow())
        print('слой удален')
        upplytable()

        packiscalc[0] = False
        checkGcodebtn()
    if plys == []:
        form.reinfcb.setEnabled(True)
        form.matrixcb.setEnabled(True)

def setplyle(cur):
    if plys[cur][0] == 0:
        plyform.bettale1.setText(str('{:.1f}'.format(plys[cur][1])))
        plyform.hlle.setText(str('{:.1f}'.format(plys[cur][5])))
        plyform.plyXminle.setText(str('{:.1f}'.format(plys[cur][3])))
        plyform.plyXmaxle.setText(str('{:.1f}'.format(plys[cur][4])))
    elif plys[cur][0] == 1:
        returncheck = False
        if plys[cur][11] == 2:
            returncheck = True
        plyform.ReturnCB2.setChecked(returncheck)
        plyform.hlle2.setText(str('{:.1f}'.format(plys[cur][5])))
        plyform.plyXminle2.setText(str('{:.1f}'.format(plys[cur][3])))
        plyform.plyXmaxle2.setText(str('{:.1f}'.format(plys[cur][4])))
def calcPack():
    global plys
    if len(plys) > 0:
        pack = [0, 0, 0, 0, 0, 0]
        Xmaxout = plys[0][4]
        Xminout = plys[0][3]
        for i in range(0, len(plys)):
            global currentPly
            currentPly[0] = i
            BufPlySet[1] = plys[0]
            setplyle(currentPly[0])
            setPlyWindow(currentPly[0])
            plyCalc(plys[currentPly[0]][0])
            OkPlySet()

            if Xminout > plys[i][3]-plys[i][13]:
                Xminout = plys[i][3]-plys[i][13]
            if Xmaxout < plys[i][4]+plys[i][13]:
                Xmaxout = plys[i][4]+plys[i][13]

            pack[0] = pack[0] + plys[i][7]
            pack[1] = pack[1] + plys[i][8]
            pack[2] = pack[2] + plys[i][9]
            pack[3] = pack[3] + plys[i][10]
            pack[4] = pack[4] + plys[i][11]

        getSupplyGcode()

        pack[5] = glWidget.maxthick

        form.Xminoutlb.setText(str('Xmin с вылетом: ' + '{:.1f}'.format(Xminout)) + ' мм.')
        form.Xmaxoutlb.setText(str('Xmax с вылетом: ' + '{:.1f}'.format(Xmaxout)) + ' мм.')

        Lwork = Xmaxout-Xminout
        form.Lworklb.setText(str('Рабочая длина: ' + '{:.1f}'.format(Lwork)) + ' мм.')


        pack[3] = pack[3] + (len(plys)-1)*Config.moveTime + Config.startTime + Config.finishTime
        form.timelb.setText(str('Время намотки: ' + '{:.1f}'.format(pack[3])) + ' мин.')

        form.cyclelb.setText(str('Количество циклов: ' + '{:.1f}'.format(pack[4])) + ' шт.')

        form.lreinflb.setText(str('Длинна ленты: ' + '{:.1f}'.format(pack[0])) + ' м')

        form.mmatrixlb.setText(str('Масса связующего: ' + '{:.1f}'.format(pack[1])) + ' г')

        mharder = pack[1] * currentMatrix[2]/(currentMatrix[2] + 1) # Масса связующего
        form.mharderlb.setText(str('Масса отвердителя: ' + '{:.1f}'.format(mharder)) + ' г')

        form.thicklb.setText(str('Толщина пакета: ' + '{:.1f}'.format(pack[5])) + ' мм')

        form.templb.setText(str('Температура: ' + '{:.1f}'.format(currentMatrix[3]) + ' °C'))

        pack[2] = pack[2]/len(plys)*100
        form.KIMlb.setText(str('КИМ:     ' + '{:.2f}'.format(pack[2])) + ' %')
        packiscalc[0] = True
        checkGcodebtn()
        up2Dgraf()
def getSupplyGcode(): #расчет переходов, стартового, конечного блоков программы
    supplyGcodes.clear()
    Ymove = 50  # Величина отвода инструмента
    startGcode = [] #Стартовый блок программы
    startGcode.append( '(Начало стартового блока)' + '\n')
    #startGcode.append('G1 Z720' + ' F' + str('{:.0f}'.format(Speed[2])) + '\n')
    startGcode.append('G1 X' + str('{:.4f}'.format(plys[0][6])) + ' F' + str('{:.0f}'.format(Speed[0])) + '\n')
    startGcode.append('(Конец стартового блока)' + '\n')

    D = float(form.Dle.text())
    finishGcode = []  # Конечный блок программы
    finishGcode.append('G92 X0 Z0 Y0' + ' (Начало конечного блока)' + '\n')
    #finishGcode.append('G28 A0 F' + str('{:.0f}'.format(Speed[3])) + ' (Возврат суппорта по A)' + '\n')
    #finishGcode.append('G1 Z720' + ' F' + str('{:.0f}'.format(Speed[2])) + '\n')
    finishGcode.append('G1 Y-50 F' + str('{:.0f}'.format(Speed[1])) + ' (Отвод суппорта по Y)' + '\n')
    finishGcode.append('M0' + ' (Ожидание отрезки)' + '\n')
    finishGcode.append('G92 Z0' + ' (Старт вращения для смотки)' + '\n')
    Z = 360 * (0.9)/(3.1415 * D * 0.001)
    finishGcode.append('G1 ' + ' Z{:.0f}'.format(Z) + ' F' + str('{:.0f}'.format(Speed[2] * 0.8)) + '\n')
    finishGcode.append('M0' + ' (Ожидание пуска вращения)' + '\n')
    #finishGcode.append('G28 X0 F' + str('{:.0f}'.format(Speed[1])) + ' (Отвод суппорта по X)' + '\n')
    finishGcode.append('G1 Z100000' + ' F' + str('{:.0f}'.format(Speed[2] * 0.1 )) + '\n')
    finishGcode.append('M2' + ' (Конец конечного блока)' + '\n')

    if len(plys) > 1: # Расчет переходов
        for i in range(0, len(plys)-1):
            moveGcode = []
            moveGcode.append('G92 Z0' + ' (Начало перехода)' + '\n')
            startGcode.append('G1 Z720' + ' F' + str('{:.0f}'.format(Speed[2])) + '\n')
            moveGcode.append('G1 Y' + str('{:.4f}'.format(-Ymove)) + ' F' + str('{:.0f}'.format(Speed[1])) + '\n')
            moveGcode.append('G1 X' + str('{:.4f}'.format(plys[i+1][6])) + ' F' + str('{:.0f}'.format(Speed[0])) + '\n')
            moveGcode.append('G1 Y0' + ' F' + str('{:.0f}'.format(Speed[1])) + '\n')
            startGcode.append('G1 Z720' + ' F' + str('{:.0f}'.format(Speed[2]*1.5)) + '\n')
            moveGcode.append('G92 Z0' + ' (Конец перехода)' + '\n')
            supplyGcodes.append(moveGcode)
    supplyGcodes.insert(0, startGcode)
    #supplyGcodes.insert(0,[])
    supplyGcodes.append(finishGcode)
def checkPlyCalc():
    global plyiscalc
    if plyiscalc == True:
        plyform.Okbtn.setEnabled(True)
        plyform.Applybtn.setEnabled(True)
    else:
        plyform.Okbtn.setEnabled(False)
        plyform.Applybtn.setEnabled(False)
def get_code():
    global Gcode
    Gcode.clear()
    Gcode.extend(supplyGcodes[0])
    for i in range(0, len(mainGcodes)):
        Gcode.append('(Слой № ' + str(i + 1) + ' ' + strategys[plys[i][0]] + ' ) \n')
        Gcode.extend(mainGcodes[i])
        Gcode.extend(supplyGcodes[i + 1])
        Gcode_None = list(filter(lambda s: s != None, Gcode))
        Gcode = Gcode_None
        saveGcode()

gcodePath = os.getcwd()
gcodeName = 'Gcode.txt'
def saveGcode():
    global Gcode
    save_Gcode_dialog()
    text_file = open(gcodePath, "w", encoding='utf-8')
    text_file.writelines(Gcode)
    sendMess('файл успешно сохранен')
    text_file.close()
    try:
       pass
    except:
        sendMess('файл занят')
    return
def save_Gcode_dialog():
    filename, ok = QFileDialog.getSaveFileName(None,"Сохранить Gcode", "Gcode.txt", "text files (*.txt)")
    if filename:
        global gcodeName
        global gcodePath
        gcodePath = Path(filename)
        gcodeName = os.path.basename(gcodePath)
        gcodeName = os.path.splitext(gcodeName)[0]
def uplply():
    if len(plys)>1 and form.plytbl.currentRow() > 0:
        i = form.plytbl.currentRow()
        plybuf = plys[i-1]
        plys[i - 1] = plys[i]
        plys[i] = plybuf
        mainGcode = mainGcodes[i - 1]
        mainGcodes[i - 1] = mainGcodes[i]
        mainGcodes[i] = mainGcode
        print('слой перемещен')
        upplytable()
        form.plytbl.selectRow(i-1)

        packiscalc[0] = False
        checkGcodebtn()

def downply():
    if len(plys)>1 and form.plytbl.currentRow() < len(plys)-1:
        i = form.plytbl.currentRow()
        plybuf = plys[i+1]
        plys[i + 1] = plys[i]
        plys[i] = plybuf
        mainGcode = mainGcodes[i + 1]
        mainGcodes[i + 1] = mainGcodes[i]
        mainGcodes[i] = mainGcode
        print('слой перемещен')
        upplytable()
        form.plytbl.selectRow(i+1)

        packiscalc[0] = False
        checkGcodebtn()

def setply():
    if len(plys) > 0:
        showWindow()
        currentPly[0] = form.plytbl.currentRow()
        if currentPly[0] < 0:
            currentPly[0] = 0
        BufPlySet[1] = plys[0]
        setplyle(currentPly[0])
        setPlyWindow(currentPly[0])

def checkInputMainForm():
    try:
        D = float(form.Dle.text()) # начальный диаметр
    except:
        form.Dle.setText(str('{:.1f}'.format(Config.DMinLim)))
        D = Config.DMinLim
        sendMess('некорректный диаметр')

    if D > Config.DMaxLim:
        form.Dle.setText(str('{:.1f}'.format(Config.DMaxLim)))
    if D < Config.DMinLim:
        form.Dle.setText(str('{:.1f}'.format(Config.DMinLim)))

    try:
        Xmin = float(form.Xminle.text())
    except:
        form.Xminle.setText(str('{:.1f}'.format(Config.XMinLim)))
        Xmin = Config.XMinLim
        sendMess('некорректный Xmin')
    try:
        Xmax = float(form.Xmaxle.text())
    except:
        form.Xmaxle.setText(str('{:.1f}'.format(Config.XMaxLim)))
        Xmax = Config.XMaxLim
        sendMess('некорректный Xmax')

    if Xmax > Config.XMaxLim:
        form.Xmaxle.setText(str('{:.1f}'.format(Config.XMaxLim)))
    if Xmin < Config.XMinLim:
        form.Xminle.setText(str('{:.1f}'.format(Config.XMinLim)))
    if (Xmax - Xmin) <= 0:
        form.Xminle.setText(str('{:.1f}'.format(Config.XMinLim)))
    if (Xmax - Xmin) > Config.Xlim:
        form.Xminle.setText(str('{:.1f}'.format(0)))
        form.Xmaxle.setText(str('{:.1f}'.format(Config.XMaxLim)))

def checkVintTab():
    try:
        plyXmin = float(plyform.plyXminle2.text())  # координата левого края
    except:
        Xmin = float(form.Xminle.text())
        plyform.plyXminle2.setText(str('{:.1f}'.format(Xmin)))
        plyXmin = Xmin
        sendMess('некорректный Xmin')

    try:
        plyXmax = float(plyform.plyXmaxle2.text())  # координата правого края
    except:
        Xmax = float(form.Xmaxle.text())
        plyform.plyXmaxle2.setText(str('{:.1f}'.format(Xmax)))
        plyXmax = Xmax
        sendMess('некорректный Xmax')

    Xmin = float(form.Xminle.text())
    Xmax = float(form.Xmaxle.text())

    plyXmax = float(plyform.plyXmaxle2.text())  # координата правого края
    if plyXmax > Xmax:
        plyform.plyXmaxle2.setText(str('{:.1f}'.format(Xmax)))
    if plyXmin < Xmin:
        plyform.plyXminle2.setText(str('{:.1f}'.format(Xmin)))
    if (plyXmax - plyXmin) <= 0:
        plyform.plyXminle2.setText(str('{:.1f}'.format(Xmin)))

    try:
        hl = float(plyform.hlle2.text())  # смещения в лентах
    except:
        plyform.hlle2.setText(str('{:.1f}'.format(Config.hlMinLim)))
        hl = Config.hlMinLim
        sendMess('некорректное смещение в лентах')

    if hl > Config.hlMaxLim:
        plyform.hlle2.setText(str('{:.1f}'.format(Config.hlMaxLim)))
    if hl < Config.hlMinLim:
        plyform.hlle2.setText(str('{:.1f}'.format(Config.hlMinLim)))
    global plyiscalc
    plyiscalc = False
    checkPlyCalc()
def checkSpiralTab():
    try:
        betta = float(plyform.bettale1.text())  # угол намотки
    except:
        plyform.bettale1.setText(str('{:.1f}'.format(Config.bettaMinLim)))
        betta = Config.bettaMinLim
        sendMess('некорректный угол бетта')
    if betta > Config.bettaMaxLim:
        plyform.bettale1.setText(str('{:.1f}'.format(Config.bettaMaxLim)))
    if betta < Config.bettaMinLim:
        plyform.bettale1.setText(str('{:.1f}'.format(Config.bettaMinLim)))

    try:
        plyXmin = float(plyform.plyXminle.text())  # координата левого края
    except:
        Xmin = float(form.Xminle.text())
        plyform.plyXminle.setText(str('{:.1f}'.format(Xmin)))
        plyXmin = Xmin
        sendMess('некорректный Xmin')

    try:
        plyXmax = float(plyform.plyXmaxle.text())  # координата правого края
    except:
        Xmax = float(form.Xmaxle.text())
        plyform.plyXmaxle.setText(str('{:.1f}'.format(Xmax)))
        plyXmax = Xmax
        sendMess('некорректный Xmax')

    Xmin = float(form.Xminle.text())
    Xmax = float(form.Xmaxle.text())
    if plyXmax > Xmax:
        plyform.plyXmaxle.setText(str('{:.1f}'.format(Xmax)))
    if plyXmin < Xmin:
        plyform.plyXminle.setText(str('{:.1f}'.format(Xmin)))
    if (plyXmax - plyXmin) <= 0:
        plyform.plyXminle.setText(str('{:.1f}'.format(Xmin)))
        plyform.plyXmaxle.setText(str('{:.1f}'.format(Xmax)))

    L = plyXmax - plyXmin

    try:
        hl = float(plyform.hlle.text())  # смещения в лентах
    except:
        plyform.hlle.setText(str('{:.1f}'.format(Config.hlMinLim)))
        hl = Config.hlMinLim
        sendMess('некорректное смещение в лентах')

    if hl > Config.hlMaxLim:
        plyform.hlle.setText(str('{:.1f}'.format(Config.hlMaxLim)))
    if hl < Config.hlMinLim:
        plyform.hlle.setText(str('{:.1f}'.format(Config.hlMinLim)))

    try:
        BettaRing = float(plyform.BettaRingle.text())  # угол намотки на кольцевом участке
    except:
        plyform.BettaRingle.setText(str('{:.1f}'.format(Config.bettaMinLim)))
        BettaRing = 70
        sendMess('некорректный угол бетта на кольцевом участке')
    if BettaRing > Config.bettaMaxLim:
        plyform.BettaRingle.setText(str('{:.1f}'.format(Config.bettaMaxLim)))
    if BettaRing < 70:
        plyform.BettaRingle.setText(str('{:.1f}'.format(70)))

    try:
        Xring = float(plyform.Xringle.text())  # длина кольцевого участка
    except:
        plyform.Xringle.setText(str('{:.1f}'.format(10)))
        Xring = L/10
        sendMess('некорректный кольцевой участок')
    if Xring > L/2:
        plyform.Xringle.setText(str('{:.1f}'.format(L/2)))
    if Xring < 0:
        plyform.Xringle.setText(str('{:.1f}'.format(0)))

    global plyiscalc
    plyiscalc = False
    checkPlyCalc()


def checkInputPlyForm():
    checkSpiralTab()
    checkVintTab()

class saveDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Сохранение")

        QBtn = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        message = QLabel("Сохранить текущий файл?")
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

def newPrj():
    global isPrjNew
    global prjName
    global packiscalc
    if isPrjNew:
        q = saveDialog()
        if q.exec():
            savePrj()
    prjName = 'NewDWCPrj'
    form.prjlb.setText('Проект: ' + prjName)
    plys.clear()
    mainGcodes.clear()
    upplytable()
    packiscalc[0] = False
    checkPlyCalc()
def savePrj():
    getPrjList()
    global isPrjNew
    global prjList
    if isPrjNew:
        saveAsPrj()
    else:
        try:
            text_file = open(prjPath, "w")
            text_file.writelines(prjList)
            sendMess('файл успешно сохранен')
        except:
            sendMess('файл занят')
    form.prjlb.setText('Проект: ' + prjName)


def saveAsPrj():
    getPrjList()
    global prjList
    save_file_dialog()
    try:
        text_file = open(prjPath, "w")
        text_file.writelines(prjList)
        sendMess('файл успешно сохранен')
    except:
        sendMess('файл занят')
    form.prjlb.setText('Проект: ' + prjName)
    return

def save_file_dialog():
    filename, ok = QFileDialog.getSaveFileName(None,"Сохранить проект", "NewDWCPrj1.txt", "text files (*.txt)")
    if filename:
        global prjPath
        global isPrjNew
        global prjName
        prjPath = Path(filename)
        prjName = os.path.basename(prjPath)
        prjName = os.path.splitext(prjName)[0]
        isPrjNew = False

def open_file_dialog():
    filename, ok = QFileDialog.getOpenFileName(None,"Открыть проект", "DWCprj.txt", "text files (*.txt)")
    if filename:
        global prjPath
        global isPrjNew
        global prjName
        prjPath = Path(filename)
        prjName = os.path.basename(prjPath)
        prjName = os.path.splitext(prjName)[0]
        isPrjNew = False
def getPrjList():
    global prjList
    global partTypes
    global currentTypeIndex
    D = float(form.Dle.text())
    Xmin = float(form.Xminle.text())
    Xmax = float(form.Xmaxle.text())

    prjList.clear()
    prjList.append('1.Тип изделия: ' + partTypes[currentTypeIndex] + '\n')
    prjList.append('2.Наполнитель: ' + currentReinf[0] + ', '
                                     + str('{:.1f}'.format(currentReinf[1])) + ', '
                                     + str('{:.1f}'.format(currentReinf[2])) + ', '
                                     + str('{:.1f}'.format(currentReinf[3])) + '\n')
    prjList.append('3.Связующее: ' + currentMatrix[0] + ', '
                                     + str('{:.1f}'.format(currentMatrix[1])) + ', '
                                     + str('{:.1f}'.format(currentMatrix[2])) + ', '
                                     + str('{:.1f}'.format(currentMatrix[3])) + '\n')
    prjList.append('4.Начальный диаметр: ' + str('{:.1f}'.format(D)) + '\n')
    prjList.append('5.Левый край: ' + str('{:.1f}'.format(Xmin)) + '\n')
    prjList.append('6.Правый край: ' + str('{:.1f}'.format(Xmax)) + '\n')
    prjList.append('7.Таблица слоев укладки \n')
    s = ['Номер слоя', 'Стратегия', 'Угол намотки', 'Угол смещ.', 'X max', 'X min', 'Параметр']
    st = '|'
    for i in range(0, len(s)):
        st = st + '{:^13}'.format(s[i]) + '|'
    prjList.append(st + '\n')
    for i in range(0, len(st)-1):
        prjList.append('_')
    prjList.append('\n')
    for i in range(0, len(plys)):
        s1 = '|' + str('{:^13}'.format(i+1)) + '|' + str('{:^13}'.format(strategys[plys[i][0]])) + '|'
        for j in range(1, 6):
            s1 = s1 + str('{:^13.1f}'.format(plys[i][j])) + '|'
        prjList.append(s1 + '\n')

def openPrj():
    global prjList
    open_file_dialog()
    try:
        text_file = open(prjPath, "r")
        prjList = text_file.readlines()
    except:
        sendMess('файл занят')
    #print(prjList)
    sendMess(parsePrjList())

def parsePrjList():
    global prjList
    global currentTypeIndex
    global currentReinf
    global currentMatrix
    global plys
    global reinfBase
    global matrixBase
    plys.clear()
    mainGcodes.clear()
    for i in range(0, len(prjList)):
        s = prjList[i]
        s = s.replace('\n', '')
        try:
            if i == 0:
                currentTypeIndex = partTypes.index(s.split(': ')[1])
                form.typecb.setCurrentIndex(currentTypeIndex)
            elif i == 1:
                reinfName = s.split(': ')[1].split(', ')[0]
                a = True
                for reinf in reinfBase:
                    if reinf[0] == reinfName:
                        currentReinf = reinf
                        a = False
                        form.reinfcb.setCurrentIndex(reinfBase.index(reinf))
                        upCurrentReinf()
                if a:
                    sendMess('Наполнитель не найден')
                    reinfform.Namele.setText(s.split(': ')[1].split(', ')[0])
                    reinfform.hle.setText(s.split(': ')[1].split(', ')[1])
                    reinfform.thickle.setText(s.split(': ')[1].split(', ')[2])
                    reinfform.Paramle.setText(s.split(': ')[1].split(', ')[3])
                    checkInputReinfForm()
                    addReinfSet()
            elif i == 2:
                matrixName = s.split(': ')[1].split(', ')[0]
                a = True
                for matrix in matrixBase:
                    if matrix[0] == matrixName:
                        currentMatrix = matrix
                        a = False
                        form.matrixcb.setCurrentIndex(matrixBase.index(matrix))
                        upCurrentMatrix()
                if a:
                    sendMess('связующее не найдено')
                    matrixform.Namele.setText(s.split(': ')[1].split(', ')[0])
                    matrixform.Gle.setText(s.split(': ')[1].split(', ')[1])
                    matrixform.ratiole.setText(s.split(': ')[1].split(', ')[2])
                    matrixform.temple.setText(s.split(': ')[1].split(', ')[3])
                    checkInputMatrixForm()
                    addMatrixSet()
            elif i == 3:
                D = s.split(': ')[1]
                form.Dle.setText(D)
            elif i == 4:
                Xmin = s.split(': ')[1]
                form.Xminle.setText(Xmin)
            elif i == 5:
                Xmax = s.split(': ')[1]
                form.Xmaxle.setText(Xmax)
            elif i >= 9:
                fields = s.split('|')
                global stratIndex
                for strat in strategys:
                    if strat == fields[2].replace(' ', ''):
                        stratIndex[0] = strategys.index(strat)
                if stratIndex[0] == 0:
                    plyform.bettale1.setText(fields[3])
                    plyform.hlle.setText(fields[7])
                    plyform.plyXminle.setText(fields[5])
                    plyform.plyXmaxle.setText(fields[6])
                    checkInputPlyForm()
                    addply()
                    plyCalc(stratIndex[0])
                    OkPlySet()
        except:
            return 'неверный файл (строка' + str(i + 1) + ')'
            plyWindow.hide()
            matrixWindow.hide()
            reinfWindow.hide()
    calcPack()
    checkInputMainForm()
    return 'файл успешно открыт'

configList = []
configPath = os.getcwd()
configName = 'DWSConfig.txt'

def openConfig():
    global configList
    open_config_dialog()
    try:
        text_file = open(configPath, "r")
        configList = text_file.readlines()
    except:
        sendMess('файл занят')
    #print(prjList)
    sendMess(parseConfigList())

def open_config_dialog():
    filename, ok = QFileDialog.getOpenFileName(None,"Открыть конфигурацию", "DWSConfig.txt", "text files (*.txt)")
    if filename:
        global configPath
        global configName
        configPath = Path(filename)
        configName = os.path.basename(prjPath)
        configName = os.path.splitext(configName)[0]

def parseConfigList():
    global configList
    global Config
    for i in range(0, len(configList)):
        s = configList[i]
        s = s.replace('\n', '')
        try:
            print(s)
        except:
            return 'неверный файл (строка' + str(i + 1) + ')'
    calcPack()
    checkInputMainForm()
    return 'файл успешно открыт'


class GLWidget(QOGLW):

    def __init__(self, parent=None):
        self.color = 255
        self.dots = [[0.0, 200.0], [0.0, 0.0]]
        self.space = 40
        self.D = float(form.Dle.text())
        self.maxthick = 6
        self.width1 = 800
        self.height1 = 200
        self.Yint = 16
        self.Ymax = self.D/2*self.maxthick
        QOGLW.__init__(self, parent)

    def initializeGL(self):
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glShadeModel(gl.GL_FLAT)
        self.reshape()

    def reshape(self):
        gl.glViewport(0, 0, self.width(), self.height())
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluOrtho2D(0.0,self.width1, 0.0, self.height1)
    def tom(self, dots = [[], []]):
        space = self.space
        mth = self.maxthick
        R = self.D / 2
        pdots = [[], []]
        k = self.height() / self.width()
        #преобразование масштаба по x
        kb = getkb(min(self.dots[0]), max(self.dots[0]), 2* space, self.width1 - space)
        for i in range(0, len(dots[0])):
            pdots[0].append(kb[0] * dots[0][i] + kb[1])
        #преобразование масштаба по y
        kb = getkb(self.Ymax-self.Yint, self.Ymax, space, self.height1 - space)
        for i in range(0, len(dots[1])):
            pdots[1].append(kb[0] * dots[1][i] + kb[1])
        return pdots

    def tomText(self, dots = [[], []]):
        space = self.space
        mth = self.maxthick
        R = self.D / 2
        pdots = [[], []]
        #преобразование масштаба по x
        kb = getkb(min(self.dots[0]), max(self.dots[0]), 2 * space, self.width1 - space)
        for i in range(0, len(dots[0])):
            pdots[0].append(kb[0] * dots[0][i] + kb[1])
        #преобразование масштаба по y
        y0 = (max(dots[1]) + min(dots[1]))/2
        k = self.height() / self.width()
        dpx = max(pdots[0]) - min(pdots[0])
        dpy = dpx / k * 0.7
        kb = getkb(self.Ymax-self.Yint, self.Ymax, space, self.height1 - space)
        yp0 = y0 * kb[0] + kb[1]
        y1 = yp0 - dpy / 2
        y2 = yp0 + dpy / 2
        if min(dots[1]) - max(dots[1]) != 0:
            kb = getkb(min(dots[1]), max(dots[1]), y1, y2)
            for i in range(0, len(dots[1])):
                pdots[1].append(kb[0] * dots[1][i] + kb[1])
        else:
            for i in range(0, len(dots[1])):
                pdots[1].append(dots[1][i] * kb[0] + kb[1])
        return pdots

    def paintGL(self):
        pollys = self.getPlysPolly()
        CSLines = self.getAxisLines()
        moldpolly = self.getMoldPolly()
        AxisLines = CSLines[0]
        AxisText = CSLines[1]
        Grid = CSLines[2]
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(255, 255, 255, 1)
        gl.glPushMatrix()
        if len(plys) > 0 and plys != [[]]:
            for polly in pollys:
                self.drawObj(gl.GL_POLYGON, polly, [255, 0, 0], 3.0, gl.GL_LINE)
            for Axis in AxisLines:
                self.drawObj(gl.GL_LINES, Axis, [0, 0, 255], 3.0, gl.GL_LINE)
            for Axis in AxisText:
                self.drawText(gl.GL_LINE_STRIP, Axis, [0, 0, 0], 2.0, gl.GL_LINE)
            for Line in Grid:
                self.drawObj(gl.GL_LINES, Line, [100, 100, 100], 0.2, gl.GL_LINE)
            self.drawObj(gl.GL_POLYGON, moldpolly, [200, 200, 200], 1.0, gl.GL_FILL)
        gl.glPopMatrix()
        gl.glFlush()
    def drawObj(self,Obj = gl.GL_POINTS,  polly = [[],[]], Color = [0, 0, 0], lineWigth = 1.0, Mode = gl.GL_LINE):
        ppolly = self.tom(polly)
        kb = getkb(0, 255, 0.0, 1.0)
        for i in range(0, len(Color)):
            Color[i] = Color[i] * kb[0] + kb[1]
        gl.glLineWidth(lineWigth)
        gl.glColor3f(Color[0], Color[1], Color[2])
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, Mode)
        gl.glBegin(Obj)
        for i in range(0, len(ppolly[0])):
            gl.glVertex2f(ppolly[0][i], ppolly[1][i])
        gl.glEnd()

    def drawText(self,Obj = gl.GL_POINTS,  polly = [[],[]], Color = [0, 0, 0], lineWigth = 1.0, Mode = gl.GL_LINE):
        ppolly = self.tomText(polly)
        kb = getkb(0, 255, 0.0, 1.0)
        for i in range(0, len(Color)):
            Color[i] = Color[i] * kb[0] + kb[1]
        gl.glLineWidth(lineWigth)
        gl.glColor3f(Color[0], Color[1], Color[2])
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, Mode)
        gl.glBegin(Obj)
        for i in range(0, len(ppolly[0])):
            gl.glVertex2f(ppolly[0][i], ppolly[1][i])
        gl.glEnd()

    def getPlysPolly(self):
        global plys
        pollys = []
        if len(plys) > 0 and plys[0] != []:
            index = 0
            for ply in plys:
                polly = [[], []]
                if ply[0] == 0:
                    plyDots = self.getPlyCritDots(ply[3]-ply[13], ply[4]+ply[13])
                    i = 0
                    while(plyDots[i] != max(plyDots)):
                        x = plyDots[i]
                        polly[0].append(x)
                        polly[1].append(self.getY(index-1, x))
                        i = i + 1
                    polly[0].append(plyDots[i])
                    polly[1].append(self.getY(index-1, plyDots[i]))
                    i = i - 1
                    while (plyDots[i] != min(plyDots)):
                        polly[0].append(plyDots[i])
                        polly[1].append(self.getY(index, plyDots[i]))
                        i = i - 1

                if ply[0] == 1:
                    plyDots = self.getPlyCritDots(ply[3] - ply[13], ply[4] + ply[13])
                    i = 0
                    while (plyDots[i] != max(plyDots)):
                        x = plyDots[i]
                        polly[0].append(x)
                        polly[1].append(self.getY(index - 1, x))
                        i = i + 1
                    polly[0].append(plyDots[i])
                    polly[1].append(self.getY(index - 1, plyDots[i]))
                    i = i - 1
                    while (plyDots[i] != min(plyDots)):
                        polly[0].append(plyDots[i])
                        polly[1].append(self.getY(index, plyDots[i]))
                        i = i - 1

                self.maxthick = max(polly[1]) - self.D/2
                pollys.append(polly)
                index = index + 1
            return pollys
        else:
            return []
    def getPlyCritDots(self, xmin, xmax):
        plyDots = []
        for dot in self.dots[0]:
            if xmin <= dot <= xmax:
                plyDots.append(dot)
        return plyDots
    def getMoldPolly(self):
        global currentTypeIndex
        global plys
        moldPolly = [[], []]
        if len(plys) > 0 and plys != [[]]:
            if currentTypeIndex == 0:
                x1 = min(self.dots[0])
                x2 = max(self.dots[0])
                y1 = self.Ymax-self.Yint
                y2 = self.D/2
                moldPolly = [[x1, x2, x2, x1],
                             [y1, y1, y2, y2]]
        return moldPolly
    def getAxisLines(self):
        L = max(self.dots[0]) - min(self.dots[0])
        n = self.width()
        k1 = 0.001
        k2 = 0.005
        period = [int(k1*n + k2*L)*10, 2]
        k1 = 300
        k2 = 0.002
        M = -0.7 + (k1/n + k2*L)
        x1 = (int(min(self.dots[0])/period[0])-1)*period[0]
        x2 = (int(max(self.dots[0])/period[0])+1)*period[0]
        y2 = (int((self.D / 2 + self.maxthick) / period[1]) + 1) * period[1]
        self.Ymax = y2
        y1 = y2 - self.Yint
        AxisLines = [[[x1, x2], #ось x
                      [y1, y1]],
                     [[x1, x1], #ось y
                      [y1, y2]]]
        AxisText  = []
        GridLines = []
        Num = []
        item1 = [[], []]
        for i in range(0, int(x2 / period[0]) + 1):
            x = float(period[0] * i)

            if i%5 == 0:
                a = 1.5
                Num = self.getTextLines(str(int(x)), x, y1 - 2 * a, M)
            else:
                a = 0.75
            gridline  =  [[x,  x], #ось x'
                          [y1, y2]]
            item1[0].append(x)
            item1[1].append(y1)
            item1[0].append(x)
            item1[1].append(y1 - a)
            AxisText.extend(Num)
            AxisLines.append(item1)
            GridLines.append(gridline)
        item1 = [[], []]
        for i in range(1, abs(int(x1 / period[0])) + 1):
            x = float(-period[0] * i)
            if i % 5 == 0:
                a = 1.5
                Num = self.getTextLines(str(int(x)), x, y1 - 2 * a, M)
            else:
                a = 0.75
            gridline = [[x, x],  # ось x'
                        [y1, y2]]
            item1[0].append(x)
            item1[1].append(y1)
            item1[0].append(x)
            item1[1].append(y1 - a)
            AxisLines.append(item1)
            AxisText.extend(Num)
            GridLines.append(gridline)
        item2 = [[],[]]
        for i in range(0, int((y2 - y1) / period[1]) + 1):
            y = y1 + period[1] * i
            if i % 2 == 0:
                b = period[0]/2
                Num = self.getTextLines(str(int(y)), x1 - 2 * b, y, M)
            else:
                b = period[0]/4
            gridline = [[x1, x2],  # ось y'
                        [y,  y]]
            item2[0].append(x1)
            item2[1].append(y)
            item2[0].append(x1 - b)
            item2[1].append(y)
            AxisLines.append(item2)
            AxisText.extend(Num)
            GridLines.append(gridline)
        return [AxisLines, AxisText, GridLines]
    def getTextLines(self, Text = '', x0 = 0.0, y0 = 0.0, m = 1.0):
        space = 8 * m
        wigth = 5 * m
        TextLines = []
        x = wigth
        y = 0
        if Text != '':
            for char in Text:
                bChar = CharLine(char, wigth)
                for i in range(0, len(bChar[0])):
                    bChar[0][i] = bChar[0][i] + x - wigth
                TextLines.append(bChar)
                x = x + space + wigth
            x2 = (x - space - wigth) / 2
            y2 = (wigth * 2) / 2
            for i in range(0, len(TextLines)):
                for j in range(0, len(TextLines[i][0])):
                    TextLines[i][0][j] = TextLines[i][0][j] - x2 + x0
                    TextLines[i][1][j] = TextLines[i][1][j] - y2 + y0

        else: TextLines = [[[],[]], [[],[]]]
        return TextLines
    def getY(self, index, x):
        global plys
        R = self.D/2
        Y = R
        for i in range(0, index+1):
            Xmax = plys[i][4]
            Xmin = plys[i][3]
            Xout = plys[i][13]
            thick = plys[i][12]
            try:
                Xring = plys[i][14]
                Xrun = plys[i][15]
            except:
                pass
            kthick = 3
            Xmaxout = Xmax + Xout
            Xminout = Xmin - Xout

            if Xminout <= x <= Xmaxout:
                if plys[i][0] == 0:
                    X1 = Xmin + Xring
                    X2 = Xmin + Xring + Xrun
                    X3 = Xmax - Xring - Xrun
                    X4 = Xmax - Xring
                    pplyDots = [[Xminout, Xmin, X1, X2, X3, X4, Xmax, Xmaxout],
                                [0.0, kthick * thick, kthick * thick, thick, thick, kthick * thick, kthick * thick, 0.0]]

                    for j in range(0, len(pplyDots[0])-1):
                        a = pplyDots[0][j]
                        b = pplyDots[0][j+1]
                        c = pplyDots[1][j]
                        d = pplyDots[1][j+1]
                        if  a < x <= b:
                            kb = getkb(a, b, c, d)
                            dY = kb[0] * x + kb[1]
                            Y = Y + dY

                if plys[i][0] == 1:
                    pplyDots = [[Xminout, Xmin, Xmax, Xmaxout], [0.0, thick, thick, 0.0]]

                    for j in range(0, len(pplyDots[0])-1):
                        a = pplyDots[0][j]
                        b = pplyDots[0][j+1]
                        c = pplyDots[1][j]
                        d = pplyDots[1][j+1]
                        if  a < x <= b:
                            kb = getkb(a, b, c, d)
                            dY = kb[0] * x + kb[1]
                            Y = Y + dY

        return Y

glWidget = GLWidget(form.centralwidget)
form.gridLayout_2.addWidget(glWidget, 0, 0)
timer = QtCore.QTimer(window)
timer.setInterval(1000)  # period, in milliseconds
timer.timeout.connect(glWidget.update)
timer.timeout.connect(glWidget.reshape)
timer.start()

def up2Dgraf():
    global plys
    Xmaxout = plys[0][4]
    Xminout = plys[0][3]
    for i in (0, len(plys)-1):
        if Xminout > plys[i][3] - plys[i][13]:
            Xminout = plys[i][3] - plys[i][13]
        if Xmaxout < plys[i][4] + plys[i][13]:
            Xmaxout = plys[i][4] + plys[i][13]
    Lwork = Xmaxout-Xminout
    critdotsX = []
    for ply in plys:
        if ply[0] == 0:
            x = ply[3]
            if x not in critdotsX:
                critdotsX.append(x)
            x = ply[4]
            if x not in critdotsX:
                critdotsX.append(x)
            x = ply[3] - ply[13]
            if x not in critdotsX:
                critdotsX.append(x)
            x = ply[4] + ply[13]
            if x not in critdotsX:
                critdotsX.append(x)
            x = ply[3] + ply[14]
            if x not in critdotsX:
                critdotsX.append(x)
            x = ply[4] - ply[14]
            if x not in critdotsX:
                critdotsX.append(x)
            x = ply[3] + ply[14] + ply[15]
            if x not in critdotsX:
                critdotsX.append(x)
            x = ply[4] - ply[14] - ply[15]
            if x not in critdotsX:
                critdotsX.append(x)

        if ply[0] == 1:
            x = ply[3]
            if x not in critdotsX:
                critdotsX.append(x)
            x = ply[4]
            if x not in critdotsX:
                critdotsX.append(x)
            x = ply[3] - ply[13]
            if x not in critdotsX:
                critdotsX.append(x)
            x = ply[4] + ply[13]
            if x not in critdotsX:
                critdotsX.append(x)
    critdotsX.sort()
    critdotsY = critdotsX.copy()
    glWidget.D = float(form.Dle.text())
    for i in range(0, len(critdotsY)):
        critdotsY[i] = 0.0
    glWidget.dots = [critdotsX, critdotsY]
    glWidget.update()

form.addplybtn.clicked.connect(addply)
form.delplybtn.clicked.connect(delply)
form.setplybtn.clicked.connect(setply)
form.upplybtn.clicked.connect(uplply)
form.downplybtn.clicked.connect(downply)
form.Gcodebtn.clicked.connect(get_code)
form.calcbtn.clicked.connect(calcPack)

form.reinftoolbtn.clicked.connect(setReinfWindow)
form.matrixtoolbtn.clicked.connect(setMatrixWindow)

form.Dle.editingFinished.connect(checkInputMainForm)
form.Xmaxle.editingFinished.connect(checkInputMainForm)
form.Xminle.editingFinished.connect(checkInputMainForm)

form.reinfcb.activated.connect(upCurrentReinf)
form.matrixcb.activated.connect(upCurrentMatrix)
form.typecb.activated.connect(upCurrentType)

form.menuFile.actions()[0].triggered.connect(newPrj)
form.menuFile.actions()[1].triggered.connect(openPrj)
form.menuFile.actions()[2].triggered.connect(savePrj)
form.menuFile.actions()[3].triggered.connect(saveAsPrj)

form.menuSet.actions()[0].triggered.connect(openConfig)

#plyform.finnished.connect()
plyform.plycalcbtn.clicked.connect(plyCalcbtn)
plyform.Cancelbtn.clicked.connect(canselPlySet)
plyform.Applybtn.clicked.connect(applyPlySet)
plyform.Okbtn.clicked.connect(OkPlySet)

plyform.bettale1.editingFinished.connect(checkInputPlyForm)
plyform.hlle.editingFinished.connect(checkInputPlyForm)
plyform.plyXminle.editingFinished.connect(checkInputPlyForm)
plyform.plyXmaxle.editingFinished.connect(checkInputPlyForm)
plyform.hlle2.editingFinished.connect(checkInputPlyForm)
plyform.plyXminle2.editingFinished.connect(checkInputPlyForm)
plyform.plyXmaxle2.editingFinished.connect(checkInputPlyForm)
plyform.Xringle.editingFinished.connect(checkInputPlyForm)
plyform.BettaRingle.editingFinished.connect(checkInputPlyForm)

reinfform.reinftbl.cellClicked.connect(setReinfLe)
reinfform.cancelbtn.clicked.connect(canselReinfSet)
reinfform.okbtn.clicked.connect(okReinfSet)
reinfform.addbtn.clicked.connect(addReinfSet)
reinfform.delbtn.clicked.connect(delReinfSet)

reinfform.hle.editingFinished.connect(checkInputReinfForm)
reinfform.thickle.editingFinished.connect(checkInputReinfForm)
reinfform.Paramle.editingFinished.connect(checkInputReinfForm)

matrixform.matrixtbl.cellClicked.connect(setMatrixLe)
matrixform.cancelbtn.clicked.connect(canselMatrixSet)
matrixform.okbtn.clicked.connect(okMatrixSet)
matrixform.addbtn.clicked.connect(addMatrixSet)
matrixform.delbtn.clicked.connect(delMatrixSet)

matrixform.Gle.editingFinished.connect(checkInputMatrixForm)
matrixform.ratiole.editingFinished.connect(checkInputMatrixForm)
matrixform.temple.editingFinished.connect(checkInputMatrixForm)


app.exec()