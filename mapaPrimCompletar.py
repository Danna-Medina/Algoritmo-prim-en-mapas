import cv2
import numpy as np
from math import sqrt

def isInTheList(elemento,arreglo):
    for i in arreglo:
        if np.array_equal(i,elemento):
            return True 
    return False

def algo_prim():
    G = []
    visitar = []
    visitar.append(verticesConectados[0])
    while len(visitar)!=len(verticesConectados):
        nuevo=[]
        for a in visitar:
            for arista in aristas:
                if np.array_equal(a,arista[0]) and (not isInTheList(arista[1],visitar)):
                    nuevo.append([a,arista[1],arista[2]])
                elif np.array_equal(a,arista[1]) and not isInTheList(arista[0],visitar):
                    nuevo.append([a,arista[0],arista[2]])
        aris = sorted(nuevo,key=lambda element:element[2])[0]
        G.append(aris)
        
        visitar.append(aris[1])
        
    for arista in G:
        cv2.line(mapa, tuple(arista[0]), tuple(arista[5]), (180,100,70), 1)
        
    for vert in verticesConectados:
        cv2.circle(mapa,(vert[0], vert[1]), 5, (128,0,128), -1)
        
    cv2.imshow('resultado',mapa)
    cv2.imwrite('resultado.png',mapa)

while(True):
    mapa=cv2.imread('mapa3.png')
    gray = cv2.cvtColor(mapa,cv2.COLOR_BGR2GRAY)

    ret,th1 = cv2.threshold(gray,254,255,cv2.THRESH_BINARY)

    kernel = np.ones((11,11), np.uint8) 

    th1 = cv2.dilate(th1,kernel,1)
    kernel = np.ones((11,11), np.uint8)
    th1 = cv2.erode(th1,kernel,1)
    th1 = cv2.GaussianBlur(th1,(5,5),cv2.BORDER_DEFAULT) 
    dst = cv2.cornerHarris(th1,2,3,0.05)
    ret, dst = cv2.threshold(dst,0.04*dst.max(),255,0)
    dst = np.uint8(dst)
    ret,th2 = cv2.threshold(th1,235,255,cv2.THRESH_BINARY)
    th2 = cv2.dilate(th2,kernel,1)

    th2 = cv2.cvtColor(th2,cv2.COLOR_GRAY2BGR)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst,30, cv2.CV_32S)
    vertices=np.int0(centroids)

    aux1=vertices
    aux2=vertices
    verticesConectados=[]
    aristas=[]
    
    for h in range(len(aux1)):
        i=aux1[h]
        for k in range(h,len(aux2)):
            j=aux2[k]
            if not (i==j).all():
                    medio = (i+j)/2
                    c1 = (medio+j)/2
                    c2 = (i+medio)/2
                    p1 = (j+c1)/2
                    p2 = (i+c1)/2
                    p3 = (medio+c1)/2
                    p4 = (medio+c2)/2
                    p5 = (j+c2)/2
                    p6 = (i+c2)/2
                    
                    if(th2[int(medio[1])][int(medio[0])]==[255,255,255]).all() and (th2[int(c1[1])][int(c1[0])]==[255,255,255]).all() and \
                    (th2[int(c2[1])][int(c2[0])]==[255,255,255]).all() and (th2[int(p1[1])][int(p1[0])]==[255,255,255]).all() and \
                    (th2[int(p2[1])][int(p2[0])]==[255,255,255]).all() and (th2[int(p3[1])][int(p3[0])]==[255,255,255]).all() and \
                    (th2[int(p4[1])][int(p4[0])]==[255,255,255]).all() and (th2[int(p5[1])][int(p5[0])]==[255,255,255]).all() and \
                    (th2[int(p6[1])][int(p6[0])]==[255,255,255]).all():
                        Peso_Arista = int(sqrt((i[0] - j[0])**2 + (i[1] - j[1])**2))
                        aristas.append([i,j,Peso_Arista])
                        
                        if not isInTheList(i, verticesConectados):
                            verticesConectados.append(i)
                        if not isInTheList(j, verticesConectados):
                            verticesConectados.append(j)
    algo_prim()
    if cv2.waitKey(1) & 0xFF==ord('a'):
                break

