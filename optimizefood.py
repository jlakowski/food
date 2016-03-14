import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def nutritionMax(targetvec, veganmode):
    with open('calories.csv', 'rb') as f:
        reader = csv.reader(f)
        caldata = list(reader)
    meatwords = ['pork', 'beef', 'sausage', 'chicken', 'clams', 'oysters', 'cod','salmon','herring', 'haddock', 'halibut', 'fish', 'crab','lobster','gelatin', 'lamb','deer', 'duck', 'tuna', 'turkey','milk', 'veal', 'trout', 'salami', 'sardines', 'flounder', 'shrimp', 'perch', 'cheese', 'scallops', 'braunschweiger', 'bologna']
    meatind = []

    #veganmode = True
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

    #caldens = sorted(caldata, key=lambda x: x[7],reverse=True)
    #protdens = sorted(caldata, key=lambda x: x[8],reverse=True)
    #carbdens = sorted(caldata, key=lambda x: x[9],reverse=True)

    #create histograms for these
    caldvals = cdnpt[7][2:].astype(float)
    protdval = cdnpt[8][2:].astype(float)
    carbdval = cdnpt[9][2:].astype(float)

    A = np.array([caldvals, protdval, carbdval])
    At= np.transpose(A)

    pdm = protdval.mean()
    cdm = caldvals.mean()
    cbdm = carbdval.mean()
    #this is the targer vector, plane, for (caldensity, protdensity, carbdensity)
    #b = np.array([cdm,pdm,cbdm])
    b = np.asarray(targetvec)
    LHS = np.dot(At, A) #this is singular
    # so we're going to do this
    #http://math.stackexchange.com/questions/381600/singular-matrix-problem
    #https://see.stanford.edu/materials/lsoeldsee263/08-min-norm.pdf
    # You can also use a QR
    # xb = At (A At)^-1 b
    MID  = np.linalg.inv(np.dot(A, At))
    L = np.dot(At, MID)

    xb = np.dot(L,b) #this should be the weighting to get us near the plane

    return [caldata, xb]

def plotResults(caldata):
    cdnp = np.asarray(caldata)
    cdnpt = np.transpose(cdnp)

    #caldens = sorted(caldata, key=lambda x: x[7],reverse=True)
    #protdens = sorted(caldata, key=lambda x: x[8],reverse=True)
    #carbdens = sorted(caldata, key=lambda x: x[9],reverse=True)

    #create histograms for these
    caldvals = cdnpt[7][2:].astype(float)
    protdval = cdnpt[8][2:].astype(float)
    carbdval = cdnpt[9][2:].astype(float)

    caldvalshist = plt.hist(caldvals)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(caldvals, protdval, carbdval)

    ax.set_xlabel('Calorie Density (kCal/g)')
    ax.set_ylabel('Protien Density, (g_(protien)/g')
    ax.set_zlabel('Carb Density (g_(carb)/g')
    
    plt.show()

def main():
    plt.close("all")
    caldata, xb = nutritionMax([1.0,1.0,1.0], veganmode=True)
    resmat = []
    for i in range(len(xb)):
        resmat.append([caldata[i+2][0], xb[i]])
    rmsort = sorted(resmat, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    main()
