import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy import arange, sin, pi,cos
from scipy import signal
from random import *
import scipy.special as special
import random
from scipy.io import wavfile


rate,data=scipy.io.wavfile.read("soundfile1_lab3.wav")  #read wav file

plt.figure(1)   #a
time=np.linspace(0,7,329413)    #create timeline equal to the number of samples
plt.plot(time,data)     #plot wav signal in time
plt.grid(True)
plt.title("Wav signal plot") #set title
plt.xlabel("Time(sec)")     #set x label
plt.ylabel("Amplitude")     #set y label


y1=data
x1=time
bit=8 #Number of bits 
kati=[] #List to store binary representation of the numbers from 0 to 2**bit-1.
for i in range(2**bit):
    kati.append(np.binary_repr(i,8)) #Use function binary_repr with 2 arguments to return a string of the binary representation of the numbers with length equal to 8.
Nsamples=len(y1)
quantised_out=np.zeros(Nsamples) #Create a list of zeros with length equal to length of y1(t).
de=2*max(y1)/float(2**bit)+1e-14 #Set Delta which is the quantization step according to the known type. We add a very small number in order to fix some anomalys.

for i in range(len(y1)):
    quantised_out[i]=de*(np.floor((y1[i]/(de)))+0.5) #Quantise the original signal y1(t) according to the Mid-riser quantiser rule.

plt.figure(2)  #b
plt.step(x1,quantised_out,where='mid') #Show the output of the quantiser.
plt.yticks(np.arange(min(quantised_out),max(quantised_out)+1e-14,2*abs(max(y1)-min(y1))/float(2**(bit+1))),kati)#Set y axis tickets equal to kati
plt.xlabel('Time(sec)')
plt.ylabel('Quantised levels in N.B.C.')
plt.title('Mid-riser quantisation of Wav file')
plt.grid(True)



str1=[] #c
kati1=np.arange(min(quantised_out),max(quantised_out)+1e-14,2*abs(max(y1)-min(y1))/float(2**(bit+1)))
for i in range(len(quantised_out)): #Use this double for loop in order to find first in which quantization level each point of the quantizer output belongs according to the smallest distance.
    for j in range(len(kati1)):
        if j==0:
            min_dinst=abs(quantised_out[i]-kati1[j])
            min_j=0
        else:            
            dinst=abs(quantised_out[i]-kati1[j])
            if(dinst<min_dinst):
                min_dinst=abs(quantised_out[i]-kati1[j])
                min_j=j
    str1.append(kati[min_j])    

nrz=[]
for i in str1:  #Use this double for loop to take the str1 string and transfrom it into bit stream. For each 0 we store to nrz 0 and for each 1 we store 1.
    for j in range(8):        
        if (int(i[j])==0):            
            nrz.append(0)
        else:
            nrz.append(1)
midriserbin=[]
for i in range(0,len(nrz),2):    
    midriserbin.append([nrz[i], nrz[i+1]])#make new list of two consecutive bits of nrz as its elements

qpsk1=np.dot(midriserbin, [2,1])
qpsk1=(qpsk1+(qpsk1==2)-(qpsk1==3)) * np.pi / 2 #make qpsk modulation using Gray code
x=np.cos(qpsk1)
y=np.sin(qpsk1)

EBN0=[5.0,15.0]#d
Es=1
noisereal=x+np.random.normal(0, np.sqrt(Es / (10**(EBN0[0]/10.0))/2), qpsk1.shape[0])    #create and add AWGN to signal as complex variable
noiseimag=y+np.random.normal(0, np.sqrt(Es / (10**(EBN0[0]/10.0))/2), qpsk1.shape[0]) 
noisereal1=x+np.random.normal(0, np.sqrt(Es / (10**(EBN0[1]/10.0))/2), qpsk1.shape[0])
noiseimag1=y+np.random.normal(0, np.sqrt(Es / (10**(EBN0[1]/10.0))/2), qpsk1.shape[0]) 

Bin=nrz

gray_bin=['00', '10', '11', '01']   #create str array of all possible 2bits combinations
index=arange(1,5,1) #quadratic(epeidi exoume qpsk)+1=4+1=5 
kati3=[]
Es=1 #stathera exartatai apo A,Tb
for j in index:
    angle=2*pi*(j-1)/4  #dimiourgo oles tis dinates gonies
    kati3.append([np.sqrt(Es), angle+pi/4])  #idaniko kima exartatai apo tis parapano gonies se morfi [platos,fasi]

qpsk=[]
for i in ((np.array(Bin)).reshape(len(Bin)/len(gray_bin[0]),len(gray_bin[0]))): #ftiaxno to qpsk se kodikopoiisi 1,2,3,4
    qpsk.append(str(gray_bin.index(''.join(map(str, i)))+1))    #analoga me tin fasi pou exei to kathe simio pairnei kai enan arithmo-tautotita

kati4=[]
for i in qpsk:
    kati4.append(kati3[int(i[len(i)-1:])-1])    #ftiaxno to kima qpsk apo to idaniko

a=np.sqrt(Es) 
gonia=[None]*5
for j in arange(1,5,1): #ftiaxno pinaka dinaton gonion apo gnosto tipo
    gonia[j]=2*pi*(j-1)/4+pi/4 #+pi/quad epeidi exo qpsk

fig=plt.figure(3)#e
ax=fig.add_subplot(111)
after_noise_real=[]
after_noise_im=[]
for j in range(len(kati4)):
    after_noise_real.append(kati4[j][0]*cos(kati4[j][1])+np.random.normal(0, (np.sqrt((Es/(10**(EBN0[0]/10)))/2)), 1)) #+noise
    after_noise_im.append(kati4[j][0] *sin(kati4[j][1])+np.random.normal(0, (np.sqrt((Es/(10**(EBN0[0]/10)))/2)), 1))
plt.scatter(after_noise_real, after_noise_im,s=20) #plot simia meta tin prosthiki thoribou se zeugoi (real,imagine) gia EBN0=5 dB 
plt.title("QPSK Constellation Diagram EBN0=5.0 dB")    
plt.grid(True)

fig=plt.figure(4)
ax=fig.add_subplot(111)
after_noise_real=[]
after_noise_im=[]
for j in range(len(kati4)):
    after_noise_real.append(kati4[j][0]*cos(kati4[j][1])+np.random.normal(0, (np.sqrt((Es/(10**(EBN0[1]/10)))/2)), 1)) #+noise
    after_noise_im.append(kati4[j][0]*sin(kati4[j][1])+np.random.normal(0, (np.sqrt((Es/(10**(EBN0[1]/10)))/2)), 1))
plt.scatter(after_noise_real, after_noise_im,s=20) #plot simia meta tin prosthiki thoribou se zeugoi (real,imagine) gia EBN0=15 dB 
plt.title("QPSK Constellation Diagram EBN0=15.0 dB")    
plt.grid(True)

ber=[]  #st
Bin=nrz
qpsk=[]
for i in ((np.array(Bin)).reshape(len(Bin)/len(gray_bin[0]),len(gray_bin[0]))): #omoia me proigoumenos ta diamorfono se qpsk simfona me tin 'tautotita' tous
    qpsk.append(str(gray_bin.index(''.join(map(str, i)))+1))

megalo_kima=[]
for i in qpsk:
    megalo_kima.append(kati3[int(i[len(i)-1:])-1])  #omoia me prin 'apokodikopoio' simfona me tin tautotita tous

for i in EBN0:
    count=0
    
    for j in range(len(megalo_kima)):
        after_noise_real1=megalo_kima[j][0]*cos(megalo_kima[j][1])+np.random.normal(0, (np.sqrt((Es/(10**(1.0*i/10)))/2)), 1) # add noise real and imaginary
        after_noise_im1=megalo_kima[j][0]*sin(megalo_kima[j][1])+np.random.normal(0, (np.sqrt((Es/(10**(1.0*i/10)))/2)), 1) 
        if ((after_noise_real1>=0) and (after_noise_im1>=0)):   #kodikopoio simfona me ta prokiptonta bits meta tin prosthiki thoribou dinontas 'tautotites' 1...4
            al="1"
        elif ((after_noise_real1<0) and (after_noise_im1>=0)):
            al="2"
        elif ((after_noise_real1<0) and (after_noise_im1<0)):
            al="3"
        else:
            al="4"
        if (al!=qpsk[j]):   #osa proekipsan me diaforetikes tautotites apo ta arxika prin tin prosthiki thoribou ta theoro lathos kai ta prostheto se enan metriti
            count += 1
    ber.append(1.0*count/len(megalo_kima))   #ektimisi BER apo to pososto lathon pros ta sinolika bit

BER_theo1=1/2.0*special.erfc(np.sqrt((2.0/np.sqrt(2))*10**(EBN0[0]/10.0)))-1/8.0*special.erfc(np.sqrt((2.0/np.sqrt(2))*10**(EBN0[0]/10.0)))**2  #calculate therotical according to known type
print "Theoretical BER =",BER_theo1,"for EBN0=5.0 dB"
print "Experimental BER =",ber[0],"for EBN0=5.0 dB"
BER_theo2=1/2.0*special.erfc(np.sqrt((2.0/np.sqrt(2))*10**(EBN0[1]/10.0)))-1/8.0*special.erfc(np.sqrt((2.0/np.sqrt(2))*10**(EBN0[1]/10.0)))**2
print "Theoretical BER =",BER_theo2,"for EBN0=15.0 dB"
print "Experimental BER =",ber[1],"for EBN0=15.0 dB"

demodqpsk=abs((noisereal<noiseimag)*(-1.0*pi/2)+(noisereal<-noiseimag)*3*pi/2)  
bindemodqpsk=(demodqpsk/pi*2).round() #z #bindemodqpsk has all 4 possible phases
bindemodqpsk=bindemodqpsk + (bindemodqpsk==2)-(bindemodqpsk==3) #Fix Gray 
final=np.dot(bindemodqpsk.reshape((bindemodqpsk.shape[0]/4,4)), [4**3, 4**2, 4**1, 1]) #Return to quantised signal with 256 levels
scipy.io.wavfile.write("After noise 5dB.wav", rate, final.astype(np.uint8)) #Create new signal of the old one with AWGN using the sampling rate

demodqpsk1=abs((noisereal1<noiseimag1)*(-1.0*pi/2)+(noisereal1<-noiseimag1)*3*pi/2)  
bindemodqpsk=(demodqpsk1/pi*2).round() 
bindemodqpsk=bindemodqpsk+(bindemodqpsk==2)-(bindemodqpsk==3) 
final=np.dot(bindemodqpsk.reshape((bindemodqpsk.shape[0]/4, 4)), [4**3, 4**2, 4**1, 1]) 
scipy.io.wavfile.write("After noise 15dB.wav", rate, final.astype(np.uint8)) 

plt.show()
