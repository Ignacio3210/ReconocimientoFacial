import cv2 as cv
import os
import numpy as np
from time import time
dataruta='D:/Cursos/Python/Data'
listaData=os.listdir(dataruta)
ids=[]
rostrosData=[]
id=0
for fila in listaData:
    rutaCompleta=dataruta+'/'+ fila
    for archivo in os.listdir(rutaCompleta):
        ids.append(id)
        rostrosData.append(cv.imread(rutaCompleta+'/'+archivo,0))
        # imagenes=cv.imread(rutaCompleta+'/'+archivo,0))
        print('Imagenes:', fila + '/'+archivo)
    id+=1
entrenamientoModelo1=cv.face.EigenFaceRecognizer_create()
tiempoInicial=time()
print("Iniciando Entrenamiento...")
entrenamientoModelo1.train(rostrosData,np.array(ids))
tiempoFinal=time()
tiempoTotal=tiempoFinal-tiempoInicial
tiempoTotal=tiempoTotal/60
entrenamientoModelo1.write('EntrenamientoEigenFaceRecognizer.xml')
print(tiempoTotal)