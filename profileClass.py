from sklearn.neural_network import MLPClassifier
from sklearn import svm
from scipy.stats import multivariate_normal
from sklearn.preprocessing import MaxAbsScaler
import scalogram
import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import sys
import os
import warnings
import argparse
warnings.filterwarnings('ignore')

def waitforEnter(fstop=True):
    if fstop:
        if sys.version_info[0] == 2:
            raw_input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")
            

## -- 1 -- ##
def plot4Classes(data1,name1,data2,name2,data3,name3,data4,name4):
    plt.subplot(4,1,1)
    plt.plot(data1)
    plt.title(name1)
    plt.subplot(4,1,2)
    plt.plot(data2)
    plt.title(name2)
    plt.subplot(4,1,3)
    plt.plot(data3)
    plt.title(name3)
    plt.subplot(4,1,4)
    plt.plot(data4)
    plt.title(name4)
    plt.show()
    waitforEnter()

def slidingObsWindow(data,lengthObsWindow,slidingValue,basename):
    iobs=0
    nSamples,nMetrics=data.shape
    obsData=np.zeros((0,lengthObsWindow,nMetrics))
    for s in np.arange(lengthObsWindow,nSamples,slidingValue):
        subdata=data[s-lengthObsWindow:s,:]
        obsData=np.insert(obsData,iobs,subdata,axis=0)
        
        obsFname="{}_obs{}_w{}.dat".format(basename,iobs,lengthObsWindow)
        iobs+=1
        np.savetxt(obsFname,subdata,fmt='%d')
               
    return obsData # 3D arrays (obs, sample, metric)

def extractStats(data):
    nSamp,nCols=data.shape

    M1=np.mean(data,axis=0)
    Md1=np.median(data,axis=0)
    Std1=np.std(data,axis=0)
    S1=stats.skew(data)
    #K1=stats.kurtosis(data)
    p=[75,90,95,98]
    Pr1=np.array(np.percentile(data,p,axis=0)).T.flatten()
        
    features=np.hstack((M1,Md1,Std1,S1,Pr1))
    # print("features")
    # print(features)
    return(features)

def extratctSilenceActivity(data,threshold=0):
    if(data[0]<=threshold):
        s=[1]
        a=[]
    else:
        s=[]
        a=[1]
    for i in range(1,len(data)):
        if(data[i-1]>threshold and data[i]<=threshold):
            s.append(1)
        elif(data[i-1]<=threshold and data[i]>threshold):
            a.append(1)
        elif (data[i-1]<=threshold and data[i]<=threshold):
            s[-1]+=1
        else:
            a[-1]+=1
    return(s,a)
    
def extractStatsSilenceActivity(data):
    features=[]
    nSamples,nMetrics=data.shape
    silence_features=np.array([])
    activity_features=np.array([])
    for c in range(nMetrics):
        silence,activity=extratctSilenceActivity(data[:,c],threshold=0)
        
        if len(silence)>0:
            silence_faux=np.array([len(silence),np.mean(silence),np.std(silence)])
        else:
            silence_faux=np.zeros(3)
        silence_features=np.hstack((silence_features,silence_faux))
        
        if len(activity)>0:
            activity_faux=np.array([len(activity),np.mean(activity),np.std(activity)])
        else:
            activity_faux=np.zeros(3)
        activity_features=np.hstack((activity_features,activity_faux))
            
    features=np.hstack((silence_features,activity_features))
        
    return(features)

def extractFeatures(dirname,basename,nObs,allwidths):
    for o in range(0,nObs):
        features=np.array([])
        for oW in allwidths:
            obsfilename=dirname+"/"+basename+str(o)+"_w"+str(oW)+".dat"
            print(obsfilename)
            subdata=np.loadtxt(obsfilename)[:,1:]    #Loads data and removes first column (sample index)
                
            faux=extractStats(subdata)    
            features=np.hstack((features,faux))
            
            # faux2=extractStatsSilenceActivity(subdata)
            # features=np.hstack((features,faux2))
        if o==0:
            obsFeatures=features
        else:
            obsFeatures=np.vstack((obsFeatures,features))

    return obsFeatures


def extractFeaturesSil(dirname,basename,nObs,allwidths):
    for o in range(0,nObs):
        features=np.array([])
        for oW in allwidths:
            obsfilename=dirname+"/"+basename+str(o)+"_w"+str(oW)+".dat"
            print(obsfilename)
            subdata=np.loadtxt(obsfilename)[:,1:]    #Loads data and removes first column (sample index)
                
            faux=extractStats(subdata)    
            features=np.hstack((features,faux))

            faux2=extractStatsSilenceActivity(subdata)
            features=np.hstack((features,faux2))

        if o==0:
            obsFeatures=features
        else:
            obsFeatures=np.vstack((obsFeatures,features))

    return obsFeatures

## -- 4 -- ##
def plotFeatures(features,oClass,f1index=0,f2index=1):
    nObs,nFea=features.shape
    colors=['b','g','r','y']
    for i in range(nObs):
        plt.plot(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

    plt.show()
    waitforEnter()
    
def logplotFeatures(features,oClass,f1index=0,f2index=1):
    nObs,nFea=features.shape
    colors=['b','g','r']
    for i in range(nObs):
        plt.loglog(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

    plt.show()
    waitforEnter()

def extractPeriodicityFeatures(dirname,basename,nObs,allwidths):
    for o in range(0,nObs):
        features=np.array([])
        for oW in allwidths:
            obsfilename=dirname+"/"+basename+str(o)+"_w"+str(oW)+".dat"
            print(obsfilename)
            subdata=np.loadtxt(obsfilename)[:,1]    #Loads data, only second column
            
            scales=np.arange(2,50)                  
            S,scales=scalogram.scalogramCWT(subdata,scales)   #periodogram using CWT (Morlet wavelet)
            features=np.hstack((features,S))
            
            #f,psd=signal.periodogram(subdata)      #periodogram using the Welch's method
            #features=np.hstack((features,pad))
            
            #fft=np.fft.fft(subdata)                   #periodogram using basic modulus-squared of the discrete FFT
            #psd=abs(fft)**2
            #features=np.hstack((features,psd))
            
        if o==0:
            obsFeatures=features
        else:
            obsFeatures=np.vstack((obsFeatures,features))

    return obsFeatures

## -- 11 -- ##
def distance(c,p):
    return(np.sqrt(np.sum(np.square(p-c))))

########### Main Code #############
Classes={0:'dev',1:'fin',2:'rh',3:'netflix'}
# plt.ion()
nfig=1

# Extract figures

## -- 1 -- ##
# pacotes
dev=np.loadtxt('dev.dat')[:,[1,3]]
fin=np.loadtxt('fin.dat')[:,[1,3]]
rh=np.loadtxt('rh.dat')[:,[1,3]]
netflix=np.loadtxt('netflix.dat')[:,[1,3]]

plt.figure(1)
plot4Classes(dev,'Dev',fin,'Fin',rh,'RH',netflix,'Netflix')

# bytes
dev=np.loadtxt('dev.dat')[:,[2,4]]
fin=np.loadtxt('fin.dat')[:,[2,4]]
rh=np.loadtxt('rh.dat')[:,[2,4]]
netflix=np.loadtxt('netflix.dat')[:,[2,4]]

plt.figure(2)
plot4Classes(dev,'Dev',fin,'Fin',rh,'RH',netflix,'Netflix')


# Obs Windowsf

fileInput=["dev.dat","fin.dat","rh.dat","netflix.dat"]
method=2
lengthObsWindow=60
slidingValue=10

for file in fileInput:
    data=np.loadtxt(file,dtype=int)
    fname=''.join(file.split('.')[:-1])
    dirname=fname+"_obs_s{}_m{}".format(slidingValue,method)
    os.mkdir(dirname)
    basename=dirname+"/"+fname

    print("\n\n### SLIDING Observation Windows with Length {} and Sliding {} ###".format(lengthObsWindow,slidingValue))
    obsData=slidingObsWindow(data,lengthObsWindow,slidingValue,basename)
    print(obsData)


# Extract Features

dirnames=["dev_obs_s10_m2","fin_obs_s10_m2","rh_obs_s10_m2","netflix_obs_s10_m2"]
widths = ['60']

allwidths=list(widths)
print("allwidths")
print(allwidths)

for dirname in dirnames:
    allfiles=os.listdir(dirname)
    nObs=len([f for f in allfiles if '_w{}.'.format(allwidths[0]) in f])
    lbn=allfiles[0].rfind("obs")+3
    basename=allfiles[0][:lbn]

    features=extractFeatures(dirname,basename,nObs,allwidths)

    outfilename=basename+"_features.dat"

    np.savetxt(outfilename,features,fmt='%.0f')

    print(features.shape)

#

## -- 3 -- ##
features_dev=np.loadtxt("dev_obs_features.dat")
features_fin=np.loadtxt("fin_obs_features.dat")
features_rh=np.loadtxt("rh_obs_features.dat")
features_netflix=np.loadtxt("netflix_obs_features.dat")

oClass_dev=np.ones((len(features_dev),1))*0
oClass_fin=np.ones((len(features_fin),1))*1
oClass_rh=np.ones((len(features_rh),1))*2
oClass_netflix=np.ones((len(features_netflix),1))*3

features=np.vstack((features_dev,features_fin,features_rh,features_netflix))
oClass=np.vstack((oClass_dev,oClass_fin,oClass_rh,oClass_netflix))

print('Train Stats Features Size:',features.shape)

## -- 4 -- ##
plt.figure(4)
plotFeatures(features,oClass,1,3)#0,8

# 0 - media pacotes download
# 1 - media bytes download
# 2 - media pacotes upload
# 3 - media bytes upload
# 4 - mediana pacotes download
# 5 - mediana bytes download
# 6 - mediana pacotes upload
# 7 - mediana bytes upload


# Extract features sil

for dirname in dirnames:
    allfiles=os.listdir(dirname)
    nObs=len([f for f in allfiles if '_w{}.'.format(allwidths[0]) in f])
    lbn=allfiles[0].rfind("obs")+3
    basename=allfiles[0][:lbn]

    features=extractFeaturesSil(dirname,basename,nObs,allwidths)

    outfilename=basename+"_sil_features.dat"

    np.savetxt(outfilename,features,fmt='%.0f')

    print(features.shape)

## -- 5 -- ##
features_devS=np.loadtxt("dev_obs_sil_features.dat")
features_finS=np.loadtxt("fin_obs_sil_features.dat")
features_rhS=np.loadtxt("rh_obs_sil_features.dat")
features_netflixS=np.loadtxt("netflix_obs_sil_features.dat")

featuresS=np.vstack((features_devS,features_finS,features_rhS,features_netflixS))
oClass=np.vstack((oClass_dev,oClass_fin,oClass_rh,oClass_netflix))

print('Train Silence Features Size:',featuresS.shape)
plt.figure(5)
plotFeatures(featuresS,oClass,38,50)

# 38 - silence len
# 39 - silence mean
# 40 - silence sd
# 50 - activity len
# 51 - activity mean
# 52 - activity sd


## -- 6 --## Periodicity

allwidths=list(widths)

for dirname in dirnames:
    allfiles=os.listdir(dirname)
    nObs=len([f for f in allfiles if '_w{}.'.format(allwidths[0]) in f])
    lbn=allfiles[0].rfind("obs")+3
    basename=allfiles[0][:lbn]

    features=extractPeriodicityFeatures(dirname,basename,nObs,allwidths)

    outfilename=basename+"_per_features.dat"
    np.savetxt(outfilename,features,fmt='%f')

    print(features.shape)

## -- 7 -- ##
features_devW=np.loadtxt("dev_obs_per_features.dat")
features_finW=np.loadtxt("fin_obs_per_features.dat")
features_rhW=np.loadtxt("rh_obs_per_features.dat")
features_netflixW=np.loadtxt("netflix_obs_per_features.dat")

featuresW=np.vstack((features_devW,features_finW,features_rhW,features_netflixW))
oClass=np.vstack((oClass_dev,oClass_fin,oClass_rh,oClass_netflix))

print('Train Wavelet Features Size:',featuresW.shape)
plt.figure(7)
plotFeatures(featuresW,oClass,3,6) # 6, 9 ou 10 ???

## -- 8 -- ##
#:1
percentage=0.5
pDev=int(len(features_dev)*percentage)
trainFeatures_dev=features_dev[:pDev,:]
pFin=int(len(features_fin)*percentage)
trainFeatures_fin=features_fin[:pFin,:]
pRh=int(len(features_rh)*percentage)
trainFeatures_rh=features_rh[:pRh,:]

trainFeatures=np.vstack((trainFeatures_dev,trainFeatures_fin,trainFeatures_rh))

trainFeatures_devS=features_devS[:pDev,:]
trainFeatures_finS=features_finS[:pFin,:]
trainFeatures_rhS=features_rhS[:pRh,:]

trainFeaturesS=np.vstack((trainFeatures_devS,trainFeatures_finS,trainFeatures_rhS))

trainFeatures_devW=features_devW[:pDev,:]
trainFeatures_finW=features_finW[:pFin,:]
trainFeatures_rhW=features_rhW[:pRh,:]

trainFeaturesW=np.vstack((trainFeatures_devW,trainFeatures_finW,trainFeatures_rhW))

o2trainClass=np.vstack((oClass_dev[:pDev],oClass_fin[:pFin],oClass_rh[:pRh]))
#i2trainFeatures=np.hstack((trainFeatures,trainFeaturesS,trainFeaturesW))
#i2trainFeatures=np.hstack((trainFeatures,trainFeaturesS))
i2trainFeatures=trainFeatures

#:2
percentage=0.5
pDev=int(len(features_dev)*percentage)
trainFeatures_dev=features_dev[:pDev,:]
pFin=int(len(features_fin)*percentage)
trainFeatures_fin=features_fin[:pFin,:]
pRh=int(len(features_rh)*percentage)
trainFeatures_rh=features_rh[:pRh,:]

pNetflix=int(len(features_netflix)*percentage)
trainFeatures_netflix=features_netflix[:pNetflix,:]

trainFeatures=np.vstack((trainFeatures_dev,trainFeatures_fin,trainFeatures_rh,trainFeatures_netflix))

trainFeatures_devS=features_devS[:pDev,:]
trainFeatures_finS=features_finS[:pFin,:]
trainFeatures_rhS=features_rhS[:pRh,:]

trainFeatures_netflixS=features_netflixS[:pNetflix,:]

trainFeaturesS=np.vstack((trainFeatures_devS,trainFeatures_finS,trainFeatures_rhS,trainFeatures_netflixS))


trainFeatures_devW=features_devW[:pDev,:]
trainFeatures_finW=features_finW[:pFin,:]
trainFeatures_rhW=features_rhW[:pRh,:]

trainFeatures_netflixW=features_netflixW[:pNetflix,:]

trainFeaturesW=np.vstack((trainFeatures_devW,trainFeatures_finW,trainFeatures_rhW,trainFeatures_netflixW))

o3trainClass=np.vstack((oClass_dev[:pDev],oClass_fin[:pFin],oClass_rh[:pRh],oClass_netflix[:pNetflix]))


#i3trainFeatures=np.hstack((trainFeatures,trainFeaturesS,trainFeaturesW))
#i3trainFeatures=np.hstack((trainFeatures,trainFeaturesS))
i3trainFeatures=trainFeatures

#:3
testFeatures_dev=features_dev[pDev:,:]
testFeatures_fin=features_fin[pFin:,:]
testFeatures_rh=features_rh[pRh:,:]

testFeatures_netflix=features_netflix[pNetflix:,:]

testFeatures=np.vstack((testFeatures_dev,testFeatures_fin,testFeatures_rh,testFeatures_netflix))

testFeatures_devS=features_devS[pDev:,:]
testFeatures_finS=features_finS[pFin:,:]
testFeatures_rhS=features_rhS[pRh:,:]

testFeatures_netflixS=features_netflixS[pNetflix:,:]

testFeaturesS=np.vstack((testFeatures_devS,testFeatures_finS,testFeatures_rhS,testFeatures_netflixS))


testFeatures_devW=features_devW[pDev:,:]
testFeatures_finW=features_finW[pFin:,:]
testFeatures_rhW=features_rhW[pRh:,:]

testFeatures_netflixW=features_netflixW[pNetflix:,:]

testFeaturesW=np.vstack((testFeatures_devW,testFeatures_finW,testFeatures_rhW,testFeatures_netflixW))

o3testClass=np.vstack((oClass_dev[:pDev],oClass_fin[:pFin],oClass_rh[:pRh],oClass_netflix[:pNetflix]))

#i3testFeatures=np.hstack((testFeatures,testFeaturesS,testFeaturesW))
#i3testFeatures=np.hstack((testFeatures,testFeaturesS))
i3testFeatures=testFeatures

## -- 9 -- ##
from sklearn.preprocessing import MaxAbsScaler

i2trainScaler = MaxAbsScaler().fit(i2trainFeatures)
i2trainFeaturesN=i2trainScaler.transform(i2trainFeatures)

i3trainScaler = MaxAbsScaler().fit(i3trainFeatures)  
i3trainFeaturesN=i3trainScaler.transform(i3trainFeatures)

i3AtestFeaturesN=i2trainScaler.transform(i3testFeatures)
i3CtestFeaturesN=i3trainScaler.transform(i3testFeatures)

# Transformacao de NaN em 0
i2trainFeaturesN = np.nan_to_num(i2trainFeaturesN)
i3AtestFeaturesN = np.nan_to_num(i3AtestFeaturesN)
i3trainFeaturesN = np.nan_to_num(i3trainFeaturesN)
i3CtestFeaturesN = np.nan_to_num(i3CtestFeaturesN)

print(np.mean(i2trainFeaturesN,axis=0))
print(np.std(i2trainFeaturesN,axis=0))

## -- 10 -- ##
from sklearn.decomposition import PCA

pca = PCA(n_components=3, svd_solver='full')

i2trainPCA=pca.fit(i2trainFeaturesN)
i2trainFeaturesNPCA = i2trainPCA.transform(i2trainFeaturesN)

i3trainPCA=pca.fit(i3trainFeaturesN)
i3trainFeaturesNPCA = i3trainPCA.transform(i3trainFeaturesN)

i3AtestFeaturesNPCA = i2trainPCA.transform(i3AtestFeaturesN)
i3CtestFeaturesNPCA = i3trainPCA.transform(i3CtestFeaturesN)

print(i2trainFeaturesNPCA.shape,o2trainClass.shape)
# plt.figure(8)
# plotFeatures(i2trainFeaturesNPCA,o2trainClass,0,1)

## -- 14 -- ##
from sklearn import svm

print('\n-- Anomaly Detection based on One Class Support Vector Machines (PCA Features) --')
ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear').fit(i2trainFeaturesNPCA)  
rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf').fit(i2trainFeaturesNPCA)  
poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',degree=2).fit(i2trainFeaturesNPCA)  

L1=ocsvm.predict(i3AtestFeaturesNPCA)
L2=rbf_ocsvm.predict(i3AtestFeaturesNPCA)
L3=poly_ocsvm.predict(i3AtestFeaturesNPCA)

AnomResults={-1:"Anomaly",1:"OK"}

nObsTest,nFea=i3AtestFeaturesNPCA.shape

sumDev=0
sumDevA=0
sumLinearDev=0
sumRBFDev=0
sumPolyDev=0
sumFin=0
sumFinA=0
sumLinearFin=0
sumRBFFin=0
sumPolyFin=0
sumRh=0
sumRhA=0
sumLinearRh=0
sumRBFRh=0
sumPolyRh=0
for i in range(0,231):
    # print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
    if Classes[o3testClass[i][0]] == 'dev':
        sumDev+=1
        if AnomResults[L1[i]] == 'Anomaly': sumLinearDev+=1
        if AnomResults[L2[i]] == 'Anomaly': sumRBFDev+=1
        if AnomResults[L3[i]] == 'Anomaly': sumPolyDev+=1
        if AnomResults[L1[i]] == 'Anomaly' and AnomResults[L2[i]] == 'Anomaly' and AnomResults[L3[i]] == 'Anomaly':
            sumDevA+=1
    elif Classes[o3testClass[i][0]] == 'fin':
        sumFin+=1
        if AnomResults[L1[i]] == 'Anomaly': sumLinearFin+=1
        if AnomResults[L2[i]] == 'Anomaly': sumRBFFin+=1
        if AnomResults[L3[i]] == 'Anomaly': sumPolyFin+=1
        if AnomResults[L1[i]] == 'Anomaly' and AnomResults[L2[i]] == 'Anomaly' and AnomResults[L3[i]] == 'Anomaly':
            sumFinA+=1
    elif Classes[o3testClass[i][0]] == 'rh':
        sumRh+=1
        if AnomResults[L1[i]] == 'Anomaly': sumLinearRh+=1
        if AnomResults[L2[i]] == 'Anomaly': sumRBFRh+=1
        if AnomResults[L3[i]] == 'Anomaly': sumPolyRh+=1
        if AnomResults[L1[i]] == 'Anomaly' and AnomResults[L2[i]] == 'Anomaly' and AnomResults[L3[i]] == 'Anomaly':
            sumRhA+=1

# print("With PCA")
print("Dev Anomaly")
print("Kernel Linear: ",(sumLinearDev/sumDev)*100,"%")
print("Kernel RBF: ",(sumRBFDev/sumDev)*100,"%")
print("Kernel Poly: ",(sumPolyDev/sumDev)*100,"%")
print("Dev Anomaly =",(sumDevA/sumDev)*100,"%")

print("\nFin Anomaly")
print("Kernel Linear: ",(sumLinearFin/sumFin)*100,"%")
print("Kernel RBF: ",(sumRBFFin/sumFin)*100,"%")
print("Kernel Poly: ",(sumPolyFin/sumFin)*100,"%")
print("Fin Anomaly =",(sumFinA/sumFin)*100,"%")

print("\nRh Anomaly")
print("Kernel Linear: ",(sumLinearRh/sumRh)*100,"%")
print("Kernel RBF: ",(sumRBFRh/sumRh)*100,"%")
print("Kernel Poly: ",(sumPolyRh/sumRh)*100,"%")
print("RH Anomaly =",(sumRhA/sumRh)*100,"%")

## -- 15 -- ##
from sklearn import svm

print('\n-- Anomaly Detection based on One Class Support Vector Machines --')
ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear').fit(i2trainFeaturesN)  
rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf').fit(i2trainFeaturesN)  
poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',degree=2).fit(i2trainFeaturesN)  

L1=ocsvm.predict(i3AtestFeaturesN)
L2=rbf_ocsvm.predict(i3AtestFeaturesN)
L3=poly_ocsvm.predict(i3AtestFeaturesN)

AnomResults={-1:"Anomaly",1:"OK"}

nObsTest,nFea=i3AtestFeaturesN.shape

sumDev=0
sumDevA=0
sumLinearDev=0
sumRBFDev=0
sumPolyDev=0
sumFin=0
sumFinA=0
sumLinearFin=0
sumRBFFin=0
sumPolyFin=0
sumRh=0
sumRhA=0
sumLinearRh=0
sumRBFRh=0
sumPolyRh=0
for i in range(0,231):
    # print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
    if Classes[o3testClass[i][0]] == 'dev':
        sumDev+=1
        if AnomResults[L1[i]] == 'Anomaly': sumLinearDev+=1
        if AnomResults[L2[i]] == 'Anomaly': sumRBFDev+=1
        if AnomResults[L3[i]] == 'Anomaly': sumPolyDev+=1
        if AnomResults[L1[i]] == 'Anomaly' and AnomResults[L2[i]] == 'Anomaly' and AnomResults[L3[i]] == 'Anomaly':
            sumDevA+=1
    elif Classes[o3testClass[i][0]] == 'fin':
        sumFin+=1
        if AnomResults[L1[i]] == 'Anomaly': sumLinearFin+=1
        if AnomResults[L2[i]] == 'Anomaly': sumRBFFin+=1
        if AnomResults[L3[i]] == 'Anomaly': sumPolyFin+=1
        if AnomResults[L1[i]] == 'Anomaly' and AnomResults[L2[i]] == 'Anomaly' and AnomResults[L3[i]] == 'Anomaly':
            sumFinA+=1
    elif Classes[o3testClass[i][0]] == 'rh':
        sumRh+=1
        if AnomResults[L1[i]] == 'Anomaly': sumLinearRh+=1
        if AnomResults[L2[i]] == 'Anomaly': sumRBFRh+=1
        if AnomResults[L3[i]] == 'Anomaly': sumPolyRh+=1
        if AnomResults[L1[i]] == 'Anomaly' and AnomResults[L2[i]] == 'Anomaly' and AnomResults[L3[i]] == 'Anomaly':
            sumRhA+=1

# print("Without PCA")
print("Dev Anomaly")
print("Kernel Linear: ",(sumLinearDev/sumDev)*100,"%")
print("Kernel RBF: ",(sumRBFDev/sumDev)*100,"%")
print("Kernel Poly: ",(sumPolyDev/sumDev)*100,"%")
print("Dev Anomaly =",(sumDevA/sumDev)*100,"%")

print("\nFin Anomaly")
print("Kernel Linear: ",(sumLinearFin/sumFin)*100,"%")
print("Kernel RBF: ",(sumRBFFin/sumFin)*100,"%")
print("Kernel Poly: ",(sumPolyFin/sumFin)*100,"%")
print("Fin Anomaly =",(sumFinA/sumFin)*100,"%")

print("\nRh Anomaly")
print("Kernel Linear: ",(sumLinearRh/sumRh)*100,"%")
print("Kernel RBF: ",(sumRBFRh/sumRh)*100,"%")
print("Kernel Poly: ",(sumPolyRh/sumRh)*100,"%")
print("RH Anomaly =",(sumRhA/sumRh)*100,"%")