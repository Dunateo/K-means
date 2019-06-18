# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import math as mth
import pickle
from random import sample
from collections import Counter

##calcul de distance euclidienne entre deux lignes de T
def calculDEUC(T1,T2):
    d=0
    for i in range(len(T1)):
        d+=mth.pow((T1[i]-T2[i]),2)
    d=np.sqrt(d)
    return(d)

##creation du tableau avec ces distances euclidiennes
def create_Tab_2DEUC(T):
    tabDE = np.zeros((len(T),len(T)),dtype=float)
    for i in range(len(T)):
        for h in range(len(T)):
            tabDE[i,h] = calculDEUC(T[i], T[h])
    return tabDE



##tire k centroide dans la liste des individus de façon aléatoire
def centroide_initiale(IND,k):
    nb_lign = IND.shape[0]
    nb_col = IND.shape[1]
    ##initialisation d'un tab d'indice de lignes IND
    tabC = np.zeros((k,nb_col))
    ligne_AL = sample(list(range(0,nb_lign)),k)

    for i in range(0,k):
        tabC[i,:] = IND[ligne_AL[i],:]

    return tabC

##renvoie le nouveau tableau de centroide calculer à partir du barycentre
def barycentre(IND, tabClust):
    nb_lign = C.shape[0]
    nb_col = C.shape[1]
    newC = np.zeros((nb_lign,nb_col))
    ##calcul les coordonées des individus dans chaque cluster
    for i in range(len(IND)):
        index = tabClust[i]
        newC[int(index),:] =  (IND[i,:] + newC[int(index),:])

    ##divise chaque cluster par son nombre de points
    cpt = Counter(tabClust)
    for h in range(len(newC)):
        newC[h,:] = newC[h,:]/cpt[h]

    return newC


##formation des clusters tableau à une dimension
def clusterisation(IND,C):
    tabClust = np.empty(len(IND))
    for i in range(len(IND)):
        tabClust[i] = np.argmin(np.sum((IND[i]-C)**2, axis=1 ) )
    return tabClust


##trace les clusters avec différentes couleurs en fonction du cluster
def tracerCluster(tabClust,IND):
    tabClust1 = np.zeros((len(IND),len(IND)))
    tabClust2 = np.zeros((len(IND),len(IND)))
    tabClust3 = np.zeros((len(IND),len(IND)))
    cpt1 = 0
    cpt2 = 0
    cpt3 = 0
    for i in range(len(IND)):
        if(tabClust[i] == 1):
            tabClust1[cpt1,0] = IND[i,0]
            tabClust1[cpt1,1] = IND[i,1]
            cpt1 = cpt1+1
        elif(tabClust[i] == 2):
            tabClust2[cpt2,0] = IND[i,0]
            tabClust2[cpt2,1] = IND[i,1]
            cpt2 = cpt2+1
        else:
            tabClust3[cpt3,0] = IND[i,0]
            tabClust3[cpt3,1] = IND[i,1]
            cpt3 = cpt3+1


    plt.scatter(tabClust1[:,0],tabClust1[:,1], color='green')
    plt.scatter(tabClust2[:,0],tabClust2[:,1], color='red')
    plt.scatter(tabClust3[:,0],tabClust3[:,1], color='blue')



##main
def main(k):
    ##ouverture du fichier
    IND =  pickle.load(open('LF','rb'))

    ##mise en place
    C = centroide_initiale(IND,k)
    tabClust = clusterisation(IND,C)
    tabClust1 = np.empty(len(tabClust))

    ##calcul les nouveau centroides tant qu'ils sont identiques aux précedent
    while np.array_equal(tabClust, tabClust1) == True:

        C = barycentre(IND,tabClust)
        tabClust1 = tabClust
        tabClust = clusterisation(IND,C)

    ##centroides
    print(C)
    ##chaque individus i correspond à un cluster tabClust[i]
    print(tabClust)

    ##tracé de la modélisation/résultat
    tracerCluster(tabClust, IND)
    plt.scatter(C[:,0],C[:,1],color='yellow')
    plt.title('HONTEUX')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal') ##repère orthonormé
    plt.show()



##lancement du programme 
main(3)
