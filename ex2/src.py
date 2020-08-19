import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy import arange, sin, pi,cos
from scipy import signal
from random import *
import scipy.special as special
import random

A=1     #Amplitude of the signal
Tb=1    #Bit duration

randBinList = lambda n: [randint(0,1) for b in range(1,n+1)]    #Lambda fucn to create random sequence
Bin=randBinList(18) #Create one random sequence of 18 bits

Bin.insert(0,0) #insert extra element in order to have proper plot
plt.figure(1)
time1=arange(0,19,1)    #create timeline
plt.step(time1,Bin) #plot the function
plt.xlabel('Time(sec)') #name x axis
plt.ylabel('Volts') #name y axis
plt.title('Random sequence')    #name whole plot
plt.xlim(-0.0001,18.0001)   #set x limits
plt.ylim(-0.05,1.05)    #set y limits
plt.grid(True)  #enable grid


############
#EROTIMA 1o#
############

bpam=[]
for i in Bin:   #Create bpam. Antikathisto 0 me -A kai 1 me A
    if i==0:
        bpam.append(-1*A)
    else:
        bpam.append(1*A)

plt.figure(2)  #a erotima
time2=arange(0,19,1)
plt.step(time2,bpam)    #plot bpam with time
plt.xlabel('Time(sec)')
plt.ylabel('Volts')
plt.title('Bpam')
plt.xlim(-0.0001,18.0001)
plt.grid(True)

plt.figure(3)  #b erotima
x1=[-A,A]   #Constellation Diagram Bpam exei mono 2 simia sta +A kai -A giati 
y1=[0,0]    # iparxoun mono 2 dinates times gia to sima
plt.scatter(x1,y1)  #plot Constellation Diagram
plt.title('Bpam Constellation Diagram')
plt.xlim(-1.5,1.5)
plt.grid(True)

EBN0=[5.0,15.0] #c erotima
noise1_real=np.random.normal(0,np.sqrt(10**(-EBN0[0]/10.0)*(A**2)*Tb/2),19) #Create WGN, 19 samples, me 0 mesi timi kai katallili diaspora
noise1_im=np.random.normal(0,np.sqrt(10**(-EBN0[0]/10.0)*(A**2)*Tb/2),19)   # to kanoume auto gia ta pragmatika kai fantastika meri
noise2_real=np.random.normal(0,np.sqrt(10**(-EBN0[1]/10.0)*(A**2)*Tb/2),19) #ton thoribon me diaforetiko EBN0 kathe fora
noise2_im=np.random.normal(0,np.sqrt(10**(-EBN0[1]/10.0)*(A**2)*Tb/2),19)
kati1=noise1_real+bpam  #Add real noise to signal
kati2=noise2_real+bpam

plt.figure(4)
plt.step(time1,kati1)   #plot new signal to time after adding noise EBN0=5 dB
plt.title('BPAM with AWGN EBN0=5.0 dB')
plt.xlabel('Time(sec)')
plt.ylabel('Volts')
plt.xlim(-0.0001,18.0001)
plt.ylim(-2,2)
plt.grid(True)

plt.figure(5)
plt.step(time1,kati2)   #plot new signal to time after adding noise EBN0=15 dB
plt.title('BPAM with AWGN EBN0=15.0 dB')
plt.xlabel('Time(sec)')
plt.ylabel('Volts')
plt.xlim(-0.0001,18.0001)
plt.ylim(-2,2)
plt.grid(True)

plt.figure(6)   #d erotima
plt.scatter(kati1,noise1_im,s=20)   #plot new signal to imaginary part of noise after adding noise EBN0=5 dB
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.title('BPAM with AWGN EBN0=5.0 dB Constellation Diagram')
plt.grid(True)

plt.figure(7)
plt.scatter(kati2,noise2_im,s=20)   #plot new signal to imaginary part of noise after adding noise EBN0=15 dB
plt.xlim(-1.5,1.5)
plt.title('BPAM with AWGN EBN0=15.0 dB Constellation Diagram')
plt.grid(True)

BER=[]  #e
for EBN0 in range(16):
    BER.append(1/2.0*special.erfc(np.sqrt((2.0/np.sqrt(2))*10**(EBN0/10.0))))   #create the ideal/theoretical BER using the known function for all EBN0 0...15 dB

plt.figure(8)
time3=arange(0,16,1)
plt.plot(time3,BER) #plot theoretical BER to EBN0 changes
plt.xlim(-0.05,16)
plt.xscale('linear') #use semilog scale in order to plot it correctly
plt.yscale('log')
plt.title('BPSK BER Theoretical Diagram for EBN0=0..15 dB')
plt.xlabel('Eb/No (dB)')
plt.ylabel('Bit Error Rate')
plt.grid(True)

N = 3000000     #have a big amount of bits
ber = [None]*16
for n in range (0,16):   
    x = 2 * (np.random.rand(N) >= 0.5) - 1  #create signal
    noise=np.random.normal(0,np.sqrt(10**(-n/10.0)/2),N)
    y = x+noise   #add WGN to signal
    y_d = 2*(y >= 0)-1
    errors = (x!=y_d).sum()     #find errors apo ta bits pou allaxan logo thoribou
    ber[n] = 1.0*errors/N   #ektimisi BER apo to pososto lathon pros ta sinolika bit

plt.figure(9)        
plt.plot(range(0,16), ber, 'bo', range(0,16), ber, 'k')     #plot experimental BER to EBN0 changes
plt.axis([0, 15, 1e-6, 0.1])
plt.xscale('linear')    #use semilog scale in order to plot it correctly
plt.yscale('log')
plt.xlabel('Eb/No (dB)')
plt.ylabel('Bit Error Rate')
plt.title('BPSK BER Experimental Diagram for EBN0=0..15 dB')
plt.grid(True)


############
#EROTIMA 2o#
############

fc=2    #carrier frequency
T=1     #T=Tb
bpsk=[]
print "Symbol Sequence for BPSK", Bin[1:] #print symbol sequence
plt.figure(10)   #b
for i in range(1,len(bpam)):    #create a bpsk signal using every bit from bpam 
    if(bpam[i]==1):             #gia bit=1 antikathistoume me cos(2pifct) kai pairnoume 100 deigmata gia ton xrono diarkeias enos bit gia na exoume ikanopoiitiko diagramma
        time4=np.linspace((i)*T,(i+1)*T,100)        
        bpsk1=[np.cos(2*pi*fc*t) for t in time4]
        bpsk=bpsk+bpsk1
    else:
        time4=np.linspace((i)*T,(i+1)*T,100)      #gia bit=0 antikathistoume me -cos(2pifct) kai pairnoume 100 deigmata antistoixa 
        bpsk1=[(-1)*np.cos(2*pi*fc*t) for t in time4]
        bpsk=bpsk+bpsk1
time4=np.linspace(0,18*T,1800)
plt.plot(time4,bpsk)    #plot bpsk in time
plt.xlim(0,18.05)
plt.title('BPSK Diagram')
plt.xlabel('Time(sec)')
plt.ylabel('Volts')
plt.grid(True)


qpsk=[]
print "Symbol Sequence for QPSK", ([(Bin[i],Bin[i+1]) for i in range(1,len(Bin),2)]) #print symbol sequence omadopoiodas ana 2 ta bits
plt.figure(11)
for i in range(1,len(bpam),2):  #ana 2 bit exetazoume olous tous dinatous sindiasmous kai dinoume analogi timi sto sima 
    time5=np.linspace(i*T,(i+2)*T,200)  #kai gia ton katallilo xrono diarkias ton 2 bit pairnoume katallila digmata simfona me to kima 
    if ((Bin[i]==0) and (Bin[i+1]==0)):     #sto opoio antistoixoun
        qpsk1=[np.cos(2*pi*fc*t) for t in time5]

    elif ((Bin[i]==0) and (Bin[i+1]==1)):
        qpsk1=[np.sin(2*pi*fc*t) for t in time5]

    elif ((Bin[i]==1) and (Bin[i+1]==1)):
        qpsk1=[(-1)*np.cos(2*pi*fc*t) for t in time5]

    else:
        qpsk1=[(-1)*np.sin(2*pi*fc*t) for t in time5]
    qpsk=qpsk+qpsk1
time5=np.linspace(0,18*T,1800)
plt.plot(time5,qpsk)    #plot qpsk in time
plt.xlim(0,18.05)
plt.title('QPSK Diagram')
plt.xlabel('Time(sec)')
plt.ylabel('Volts')
plt.grid(True)


dec=[]
for i in range(1,len(Bin),3):   #ana 3 bit dimiourgoume ton antistoixo dekadiko arithmo se enan pianaka dec
    if(Bin[i]==0 and Bin[i+1]==0):
        dec.append(Bin[i+2]+1)
    if(Bin[i]==0 and Bin[i+1]==1):
        dec.append(4-Bin[i+2])
    if(Bin[i]==1 and Bin[i+1]==1):
        dec.append(Bin[i+2]+5)
    if(Bin[i]==1 and Bin[i+1]==0):
        dec.append(8-Bin[i+2])
j=0
psk=[]
print "Symbol Sequence for 8-PSK", ([(Bin[i],Bin[i+1],Bin[i+2]) for i in range(1,len(Bin),3)]) #print symbol sequence omadopoiimena ana 3 bits
plt.figure(12)
for i in range(1,len(Bin),3):
    time6=np.linspace(i*T,(i+3)*T,300)  #se xrono diarkeias 3 bits pairnoume simfona me ton gnosto tipo tin katallili morfi simatos kai epita deigmata gia to 8-psk sima
    psk1=[(A)*(np.cos(2*pi*fc*t-((dec[j]-1)*2*pi/8))) for t in time6]   #analogos me tin timi ton 3 auton bits sto dekadiko sistima simfona me ton pinaka dec parapano
    j=j+1
    psk=psk+psk1
time6=np.linspace(0,18*T,1800)
plt.plot(time6,psk)     #plot 8-psk in time
plt.ylim(-1.05,1.05)
plt.xlim(0,18.05)
plt.title('8-PSK Diagram')
plt.xlabel('Time(sec)')
plt.ylabel('Volts')
plt.grid(True)


############
#EROTIMA 3o#
############

k=np.sqrt(2)    #a erotima
gray=np.array([1.0+1.0j, -1.0+1.0j, 1.0-1.0j, -1.0-1.0j])/k**2     #create array of all possible position for qpsk
fig=plt.figure(13)
ax=fig.add_subplot(111)     #subplot gia na parousiastoun 2 diagrammata mazi
circle1=plt.Circle((0,0),A/k,color='g',fill=False) #plot circle that passes from all 4 points
ax.add_artist(circle1)
plt.scatter(gray.real,gray.imag) #plot 4 points too
plt.xlim(-1.25,1.25)
plt.ylim(-1.25,1.25)
ax.annotate('Q', xy=(0.1,1.15),textcoords='data') #name axis with Q and I
ax.annotate('I', xy=(1.2,0.1),textcoords='data')
ax.annotate('s4(t) 10',xy=(gray[2].real,gray[2].imag+0.05),textcoords='data')   #Name points with s0...s4 simfona me tis sintetagmenes tous
ax.annotate('s0(t) 00',xy=(gray[0].real,gray[0].imag+0.05),textcoords='data')
ax.annotate('s3(t) 11',xy=(gray[3].real,gray[3].imag+0.05),textcoords='data')
ax.annotate('s1(t) 01',xy=(gray[1].real,gray[1].imag+0.05),textcoords='data')
plt.title('QPSK Constellation Diagram')
plt.grid(True)

EBN0=[5.0,15.0]     #b erotima
Bin.pop(0) #remove the extra added bit in Bin
gray_bin=['00', '10', '11', '01']   #create str array of all possible 2bits combinations
index=arange(1,5,1) #quadratic(epeidi exoume qpsk)+1=4+1=5 
kati3=[]
Es=(A**2)*Tb/2.0 #stathera exartatai apo A,Tb
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

fig=plt.figure(14)
ax=fig.add_subplot(111)
circle1=plt.Circle((0,0),1.0/np.sqrt(2),color='g',fill=False)
ax.add_artist(circle1)  #plot circle
for j in arange(1,5,1):
    plt.scatter(a*cos(gonia[j]), a*sin(gonia[j]), color='b') #plot idanika simia gia EBN0=5 dB with blue     
after_noise_real=[]
after_noise_im=[]
for j in range(len(kati4)):
    after_noise_real.append(kati4[j][0]*cos(kati4[j][1])+np.random.normal(0, (np.sqrt((2*Es/(10**(EBN0[0]/10)))/2)), 1)) #+noise
    after_noise_im.append(kati4[j][0] *sin(kati4[j][1])+np.random.normal(0, (np.sqrt((2*Es/(10**(EBN0[0]/10)))/2)), 1))
plt.scatter(after_noise_real, after_noise_im,s=20, color='r') #plot simia meta tin prosthiki thoribou se zeugoi (real,imagine) gia EBN0=5 dB with red
plt.title("QPSK Constellation Diagram EBN0=5.0 dB")    
plt.xlim(-2, 2)
plt.ylim(-1.5, 1.5)
ax.annotate('Q', xy=(0.1,1.4),textcoords='data') #name axis with Q and I
ax.annotate('I', xy=(1.45,0.1),textcoords='data')
plt.grid(True)

fig=plt.figure(15)
ax=fig.add_subplot(111)
circle1=plt.Circle((0,0),1.0/np.sqrt(2),color='g',fill=False)
ax.add_artist(circle1) #plot circle
for j in arange(1,5,1):
    plt.scatter(a*cos(gonia[j]), a*sin(gonia[j]), color='b')  #plot idanika simia gia EBN0=15 dB with blue      
after_noise_real=[]
after_noise_im=[]
for j in range(len(kati4)):
    after_noise_real.append(kati4[j][0]*cos(kati4[j][1])+np.random.normal(0, (np.sqrt((2*Es/(10**(EBN0[1]/10)))/2)), 1)) #+noise
    after_noise_im.append(kati4[j][0]*sin(kati4[j][1])+np.random.normal(0, (np.sqrt((2*Es/(10**(EBN0[1]/10)))/2)), 1))
plt.scatter(after_noise_real, after_noise_im,s=20, color='r') #plot simia meta tin prosthiki thoribou se zeugoi (real,imagine) gia EBN0=15 dB with red
plt.title("QPSK Constellation Diagram EBN0=15.0 dB")    
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
ax.annotate('Q', xy=(0.1,1.4),textcoords='data') #name axis with Q and I
ax.annotate('I', xy=(1.45,0.1),textcoords='data')
plt.grid(True)

ber=[None]*16   #c erotima
Bin=randBinList(100000)     #dimiourgo megalo digma tixaion bits
qpsk=[]
for i in ((np.array(Bin)).reshape(len(Bin)/len(gray_bin[0]),len(gray_bin[0]))): #omoia me proigoumenos ta diamorfono se qpsk simfona me tin 'tautotita' tous
    qpsk.append(str(gray_bin.index(''.join(map(str, i)))+1))

megalo_kima=[]
for i in qpsk:
    megalo_kima.append(kati3[int(i[len(i)-1:])-1])  #omoia me prin 'apokodikopoio' simfona me tin tautotita tous

for i in range(16):
    count=0
    for j in range(len(megalo_kima)):
        after_noise_real1=megalo_kima[j][0]*cos(megalo_kima[j][1])+np.random.normal(0, (np.sqrt((2*Es/(10**(1.0*i/10)))/2)), 1) # add noise real and imaginary
        after_noise_im1=megalo_kima[j][0]*sin(megalo_kima[j][1])+np.random.normal(0, (np.sqrt((2*Es/(10**(1.0*i/10)))/2)), 1) 
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
    ber[i]=1.0*count/len(megalo_kima)   #ektimisi BER apo to pososto lathon pros ta sinolika bit
plt.figure(16)
plt.plot(range(0,16), ber, 'bo', range(0,16), ber, 'k')     #plot experimental BER to EBN0 changes
plt.xscale('linear')    #use semilog scale in order to plot it correctly
plt.yscale('log')
plt.title("QPSK BER Experimental Diagram for EBN0=0..15 dB")
plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate")
plt.grid(True)

BER1=[]  
for EBN0 in range(16):
    BER1.append(1/2.0*special.erfc(np.sqrt((2.0/np.sqrt(2))*10**(EBN0/10.0)))-1/8.0*special.erfc(np.sqrt((2.0/np.sqrt(2))*10**(EBN0/10.0)))**2) #create the ideal/theoretical BER using the known function for all EBN0 0...15 dB

plt.figure(17)
time3=arange(0,16,1)
plt.plot(time3,BER1) #QPSK plot theoretical BER to EBN0 changes
#plt.plot(time3,BER) #BPSK plot theoretical BER to EBN0 changes
plt.xlim(-0.05,16)
plt.xscale('linear') #use semilog scale in order to plot it correctly
plt.yscale('log')
plt.title('QPSK BER Theoretical Diagram for EBN0=0..15 dB')
plt.xlabel('Eb/No (dB)')
plt.ylabel('Bit Error Rate')
plt.grid(True)

plt.show()
