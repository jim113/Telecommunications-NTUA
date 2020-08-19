import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy import arange, sin, pi
from scipy import signal

fm=1000 #frequency of the signal
A=1     #Amplitude of the signal
fs1=20*fm #Sampling frequency fs1
fs2=100*fm #Sampling frequency fs2
T=1/float(fm) #Calculate period of signal
sample_rate1=1/float(fs1) #Sampling rate for fs1
sample_rate2=1/float(fs2) #Sampling rate for fs1

############
#EROTIMA 1o#
############

plt.figure(1)  #a i
x1 = arange(4*float(T)/sample_rate1+0.000001) #Create timeline for y1(t) with range of 4 periods and store data to x1. We add a small number in order to have exactly 4 periods.
y1 = [ A*sin(2*pi*fm*(i/fs1)) for i in x1]  #Sampling the original signal with fs1 and store data to y1.
plt.scatter(x1/fs1, y1,s=10) #Use scatter in order to show the sampled signal. Use 's' parameter in order to small the size of the marks.
plt.xlabel('Time(sec)')    #Set x axis label to time.
plt.ylabel('y1(t)')    #Set y axis label to y1(t).
plt.title('Sampling with fs1') #Set a title to the diagram. 
plt.xlim(-0.0001,0.0041) #Set the correct limits in x axis in order to have a better view of the plotted signal.
plt.grid(True) #Activate the parameter in order to have a grid in the background; so we can estimate better signal values.

plt.figure(2)  #a ii
x2 = arange(4*float(T)/sample_rate2+0.000001) #Create timeline for y2(t)
y2 = [ A*sin(2*pi*fm *(i/fs2)) for i in x2] #Sampling with fs2.
plt.scatter(x2/fs2, y2,s=0.25)    #Show the sampled signal.
plt.xlabel('Time(sec)')
plt.ylabel('y2(t)')
plt.title('Sampling with fs2')
plt.xlim(-0.0001,0.0041)
plt.grid(True)

plt.figure(3)  #a iii
plt.scatter(x2/fs2,y2,s=0.25)  #Show both samples in the same diagram.
plt.scatter(x1/fs1,y1,s=10) 
plt.xlabel('Time(sec)')
plt.ylabel('y1(t),y2(t)')
plt.title('Both samples of sampling with fs1(Orange) and fs2(Blue)')
plt.xlim(-0.0001,0.0041)
plt.grid(True)

plt.figure(4)  #b
fs3=5*fm
x3 = arange(4*float(T)*fs3+0.000001) #Create timeline for y3(t).
y3 = [ A*sin(2*pi*fm*(i/fs3)) for i in x3] #Sampling with fs.
plt.scatter(x3/fs3, y3,s=10) #Show the sampled signal.
plt.xlabel('Time(sec)')
plt.ylabel('y3(t)')
plt.title('Sampling with fs=5fm')
plt.xlim(-0.0001,0.0041)
plt.grid(True)

plt.figure(5)  #c a i
x4 = arange(float(T)/sample_rate1+0.000001) #Create timeline for y4(t).
y4 = [ (A*sin(2*pi*fm*(i/fs1))+A*sin(2*pi*(fm+1000)*(i/fs1))) for i in x4] #Sampling z(t) with fs1. 
plt.scatter(x4/fs1, y4,s=10) #Show the sampled signal.
plt.xlabel('Time(sec)')
plt.ylabel('y4(t)')
plt.title('Sampling with fs1 for z(t)')
plt.xlim(-0.0001,0.0011)
plt.grid(True)

plt.figure(6)  #c a ii
x5 = arange(float(T)/sample_rate2+0.000001) #Create timeline for y5(t).
y5 = [ (A*sin(2*pi*fm*(i/fs2))+A*sin(2*pi*(fm+1000)*(i/fs2))) for i in x5] #Sampling z(t) with fs2.
plt.scatter(x5/fs2, y5,s=4) #Show the sampled signal.
plt.xlabel('Time(sec)')
plt.ylabel('y5(t)')
plt.title('Sampling with fs2 for z(t)')
plt.xlim(-0.0001,0.0011)
plt.grid(True)

plt.figure(7)  #c a iii
plt.scatter(x4/fs1, y4,s=10,color='green') #Show both samples in the same diagram with different colors.
plt.scatter(x5/fs2, y5,s=2,color='red') 
plt.xlabel('Time(sec)')
plt.ylabel('y1(t),y2(t)')
plt.title('Both samples of sampling for z(t) with fs1(Green) and fs2(Red)')
plt.xlim(-0.0001,0.0011)
plt.grid(True)

plt.figure(8)  #c b
x6 = arange(float(T)*fs3+0.000001) #Create timeline for y6(t).
y6 = [ (A*sin(2*pi*fm*(i/fs3))+A*sin(2*pi*(fm+1000)*(i/fs3))) for i in x6] #Sampling z(t) with fs.
plt.scatter(x6/fs3, y6,s=12) #Show the sampled signal.
plt.xlabel('Time(sec)')
plt.ylabel('y6(t)')
plt.title('Sampling with fs=5fm for z(t)')
plt.xlim(-0.0001,0.0011)
plt.grid(True)


############
#EROTIMA 2o#
############

bit=4 #Number of bits depended on A.M.
kati=[] #List to store binary representation of the numbers from 0 to 2**bit-1.
for i in range(2**bit):
    kati.append(np.binary_repr(i,4)) #Use function binary_repr with 2 arguments to return a string of the binary representation of the numbers with length equal to 4.
Nsamples=len(y1)
quantised_out=np.zeros(Nsamples) #Create a list of zeros with length equal to length of y1(t).
de=2*A/float(2**bit)+1e-14 #Set Delta which is the quantization step according to the known type. We add a very small number in order to fix some anomalys.

for i in range(len(y1)):
    quantised_out[i]=de*(np.floor((y1[i]/(de)))+0.5) #Quantise the original signal y1(t) according to the Mid-riser quantiser rule.

plt.figure(9)  #a
plt.step(x1/fs1,quantised_out,where='mid') #Show the output of the quantiser.
#plt.plot(x1/fs1, y1,color='red')   #Show the input of the quantiser with red color.
plt.xlim(-0.00005,0.00405)
plt.yticks(arange(min(quantised_out),max(quantised_out)+1e-14,2/float(2**bit)),kati)#Set y axis tickets equal to kati=('0000','0001','0010','0011','0100','0101','0110','0111','1000','1001','1010','1011','1100','1101','1110','1111')
plt.xlabel('Time(sec)')
plt.ylabel('Quantised levels in N.B.C.')
plt.title('Mid-riser quantisation of y1(t)')
plt.grid(True)

z=y1-quantised_out #b i&ii Define the quantization error as the difference of input - output of the quantiser.
s1=np.std(z[:10],ddof=1)    #Use std function in order to calculate the Standard Deviation of the first 10 samples of the error.
s2=np.std(z[:20],ddof=1)    #Use std function in order to calculate the Standard Deviation of the first 20 samples of the error.
print 'Standard deviation (tipiki apoklisi) for the first 10 samples',s1
print 'Standard deviation (tipiki apoklisi) for the first 20 samples',s2 #theortika s=(de^2/12)^(1/2)=0.0360843914


P = lambda k: np.dot(k, np.conj(k)) * 1.0 / len(k)  # b iii Define the function of calculating the Power of the input signal.
snr1=10*np.log10(P(y1[:10])/(np.var(z[:10],ddof=1)))    #Calculate the SNR in dB for the first 10 samples according to the known rule: SNR=P/s**2
snr2=10*np.log10(P(y1[:20])/(np.var(z[:20],ddof=1)))    #Calculate the SNR in dB for the first 20 samples according to the known rule: SNR=P/s**2
print 'SNR for the first 10 samples',snr1,'dB'
print 'SNR for the first 20 samples',snr2,'dB' #theortika snr=26dB

str1=[]
kati1=arange(min(quantised_out),max(quantised_out)+1e-14,2/float(2**bit)) #Create a list that has as elements the quantisation levels.
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
    str1.append(kati[min_j])    #Then we store to str1 the appropriate N.B.C. from the list named 'kati'.

nrz=[]
for i in str1:  #Use this double for loop to take the str1 string and transfrom it into bit stream. For each 0 we store to nrz -1 and for each 1 we store 1.
    for j in range(4):        
        if (int(i[j])==0):            
            nrz.append(-1)
        else:
            nrz.append(1)
nrz.insert(0,0) #In order to have the correct duration for the first bit we store to the first position of nrz a 0. The following bits after the first have the correct duration. 

plt.figure(10) #c
time1=arange(0,0.325,0.001) #Create timeline in order each bit has duration 1msec.
plt.step(time1,nrz)    #Show the bit stream.
plt.xlim(-0.00005,0.081)   #Set limits in order to show 1 period.
plt.xlabel('Time(sec)')
plt.ylabel('Volts')
plt.title('Bit stream')
plt.grid(True)


############
#EROTIMA 3o#
############

plt.figure(11) #a
k1=float(1)/30 #Calculate period of m(t)
x7 = np.linspace(0,4*k1,(100.0*fm))    #Create timeline for y7(t) for 4 periods in order to have the correct AM modulation.
y7 = [((sin(2*pi*100*fm*i))+0.25*(sin(2*pi*(30+100*fm)*(i)-pi/2.0)+sin(2*pi*(100*fm-30)*(i)+pi/2.0))) for i in x7]  #Determine the AM modulated signal according to the known rule.
plt.xlim(-0.0001,0.128) #Set limits in order to show 4 periods.
plt.plot(x7,y7) 
plt.xlabel('Time(sec)')
plt.ylabel('y7(t)')
plt.title('AM modulated signal of m(t)')
plt.grid(True)

plt.figure(12) #b
y8 = [ sin(2*pi*30*(i)) for i in x7]    #Create the original m(t).
env=signal.hilbert(y7)  #In order to AM demodulate we use Hilbert Transform.
kaf=abs(2*env)-2    #In order to recover m(t) according to the known rule we take the abs of  the double of Hilbert Transform. We subtract 2 in order to bring the signal around 0.
plt.plot(x7,kaf)   #Show AM demodulated signal.
#plt.plot(x7,y8,color='red') #Show m(t) with red color.
plt.xlabel('Time(sec)')
plt.ylabel('y7(t)')
plt.xlim(0,0.1332)
plt.title('AM demodulated signal of y7(t))')
plt.grid(True)

plt.show()
