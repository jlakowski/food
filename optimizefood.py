import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open('calories.csv', 'rb') as f:
    reader = csv.reader(f)
    caldata = list(reader)
meatwords = ['pork', 'beef', 'sausage', 'chicken', 'clams', 'oysters', 'cod','salmon','herring', 'haddock', 'halibut', 'fish', 'crab','lobster','gelatin', 'lamb','deer', 'duck', 'tuna', 'turkey','milk', 'veal', 'trout', 'salami', 'sardines', 'flounder', 'shrimp', 'perch', 'cheese', 'scallops', 'braunschweiger']
meatind = []

veganmode = True
for k in range(2, len(caldata)):
    for i in range(2,len(caldata[2])):
        caldata[k][i] = float(caldata[k][i])
    for word in meatwords:
        if(word in caldata[k][0].lower()):
            meatind.append(k)

if veganmode:
    for k in range(len(caldata), 2, -1):
        if k in meatind:
            caldata.pop(k)

cdnp = np.asarray(caldata)
cdnpt = np.transpose(cdnp)

caldens = sorted(caldata, key=lambda x: x[7],reverse=True)
protdens = sorted(caldata, key=lambda x: x[8],reverse=True)
carbdens = sorted(caldata, key=lambda x: x[9],reverse=True)

caldvals = cdnpt[7][2:].astype(float)
protdval = cdnpt[8][2:].astype(float)
carbdval = cdnpt[9][2:].astype(float)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(caldvals, protdval, carbdval)

ax.set_xlabel('Calorie Density (kCal/g)')
ax.set_ylabel('Protien Density, (g_(protien)/g')
ax.set_zlabel('Carb Density (g_(carb)/g')
#see if there are some clusters using a clustering algorithm



A = np.array([caldvals, protdval, carbdval])
At= np.transpose(A)

pdm = protdval.mean()
cdm = caldvals.mean()
cbdm = carbdval.mean()
#this is the ideal formula for (caldensity, protdensity, carbdensity)

b = np.array([cdm,0.8,cbdm])

LHS = np.dot(At, A) #this is singular
# so we're going to do this
#http://math.stackexchange.com/questions/381600/singular-matrix-problem
MID  = np.linalg.inv(np.dot(A, At))
L = np.dot(At, MID)

xb = np.dot(L,b) #this should be the weighting

resmat = []
for i in range(len(xb)):
    resmat.append([caldata[i+2][0], xb[i]])
rmsort = sorted(resmat, key=lambda x: x[1], reverse=True)


#plt.show()
