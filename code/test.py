
import numpy as np
import scipy.io as sio
import socket
import datetime
import threading
import time
import serial  
import io  
import pickle
from scipy.fftpack import fft, ifft
from multiprocessing import Process
Fs=128.0
N = 641;
T = 1.0 / Fs
f = np.linspace(0.0, 1.0/(2.0*T), N//2)

first_choise=1
second_choise=3
no_choise1=2
no_choise2=4

filename = 'finalized_model_asma6.sav'
loaded_model = pickle.load(open(filename, 'rb'))

filename =  'finalized_model_asma6sc.sav'
loaded_modelsc = pickle.load(open(filename, 'rb'))

filename = 'finalized_model_asma6svm.sav'
loaded_modelsvm = pickle.load(open(filename, 'rb'))



HOST2 = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT2 = 65432  

HOSTcpp = '127.0.0.1'  # Standard loopback interface address (localhost)
PORTcpp =  27015        # Port to listen on (non-privileged ports are > 1023)

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT =  54123        # Port to listen on (non-privileged ports are > 1023)
eeg=np.array([[],[],[],[],[],[],[],[],[],[],[],[],[],[]])
events=np.array([[0,24]])

class ref:
    def __init__(self, obj): self.obj = obj
    def get(self):    return self.obj
    def set(self, obj):      self.obj = obj
send0 = ref(0)
send1 = ref(0)
send2 = ref(0)
send3 = ref(0)
send4 = ref(0)
send5 = ref(0)

class1 = ref(0)
class2 = ref(0)
class3 = ref(0)

motor = ref(0)
direction = ref(0)
grip = ref(0)
def sendf1():
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.connect((HOST2, PORT2))
		s.send(str(motor.get()[0]).encode()) 
		time.sleep(1)
		print("m")
		print(motor.get()[0])
		s.close()


def sendf2():	
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.connect((HOST2, PORT2))
		s.send(str(direction.get()[0]).encode()) 
		time.sleep(1)
		print("d")
		print(direction.get()[0])
		s.close()

def sendf3():	
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.connect((HOST2, PORT2))
		s.send(str(grip.get()[0]).encode()) 
		time.sleep(1)
		print("g")
		print(grip.get()[0])
		s.close()

def classify():
	print("classification started")

	condition1 = events[:,1]==0
	condition2 = events[:,1]==1
	condition= [any(x) for x in zip(condition1, condition2)]
	code=np.extract(condition, events[:,0])
	print(code)
	before=eeg[6,code[1]-640:code[1]]
	mean= np.mean(eeg, axis=0)
	after=before - mean[code[1]-640:code[1]]

	y = fft(after)
	p1=2.0/N * np.abs(y[0:N//2])
	
	p1=np.array([p1])
	#X_test = loaded_modelsc.transform(p1) 
	#y_pred = loaded_modelsvm.predict(X_test)
	y_pred = loaded_model.predict(p1)
	motor.set(y_pred)
	sendf1()
	class1.set(1)

def classify2(i):
	
	if (i==0):
		condition1 = events[:,1]==2
		condition2 = events[:,1]==3
	
	else :
		condition1 = events[:,1]==4
		condition2 = events[:,1]==5
		
	condition= [any(x) for x in zip(condition1, condition2)]
	code=np.extract(condition, events[:,0])
	print(code)
	mean= np.mean(eeg, axis=0)
	before=eeg[6,code[1]-640:code[1]]
	after=before - mean[code[1]-640:code[1]]

	y = fft(after)
	p1=2.0/N * np.abs(y[0:N//2])
	
	p1=np.array([p1])
	#X_test = loaded_modelsc.transform(p1) 
	#y_pred = loaded_modelsvm.predict(X_test)
	
	y_pred = loaded_model.predict(p1)

	
	if(i==0):
		if(y_pred[0]==first_choise):
			direction.set([3])
			sendf2()
			print("direction is :" ,direction.get(),y_pred)
		if(y_pred[0]==second_choise):
			direction.set([4])
			sendf2()
			print("direction is :" ,direction.get(),y_pred)
		if(y_pred[0]==no_choise1):
			direction.set([5])
			print("direction is :" ,direction.get(),y_pred)
		if(y_pred[0]==no_choise2):
			direction.set([5])
			print("direction is :" ,direction.get(),y_pred)
		#direction.set([5])
		
		class2.set(1)	
	else:
	
		if(y_pred[0]==first_choise):
			grip.set([3])
			#sendf3()
			print("grip is :" ,grip.get(),y_pred)
		if(y_pred[0]==second_choise):
			grip.set([4])
			#sendf3()
			print("grip is :" ,grip.get(),y_pred)
		if(y_pred[0]==no_choise1):
			grip.set([5])
			print("grip is :" ,grip.get(),y_pred)
		if(y_pred[0]==no_choise2):
			grip.set([5])
			print("grip is :" ,grip.get(),y_pred)
		grip.set([4])
		sendf3()
		class3.set(1)


def cppThread():
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as c:

		c.connect((HOSTcpp, PORTcpp))
		c.send(b"should I start?")
		while(1):
		
			datacpp = c.recv(1024)
			print(repr(datacpp))
			
			if( repr(datacpp)=="b'0'"):
				send0.set(1)
			if( repr(datacpp)=="b'1'"):
				send1.set(1)
				while(class1.get()!=1):
					i=0
				class1.set(0)
				#print(motor.get())
				c.send(b"00"+str(motor.get()[0]).encode())
				#c.send(b"00")
			
			if( repr(datacpp)=="b'2'"):
				send2.set(1)
			if( repr(datacpp)=="b'3'"):
				send3.set(1)
				while(class2.get()!=1):
					i=0
				class2.set(0)
				c.send(b"00"+str(direction.get()[0]).encode())
				#c.send(b"00")
			if( repr(datacpp)=="b'4'"):
				send4.set(1)
			if( repr(datacpp)=="b'5'"):
				send5.set(1)
				while(class3.get()!=1):
					i=0
				class3.set(0)
				c.send(b"00"+str(grip.get()[0]).encode())
				#c.send(b"00")
	

threading.Thread(target=cppThread).start()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
	s.connect((HOST, PORT))
	s.send(b"\r\n")
	while (1):
			
			data = s.recv(1024)
			my_string=repr(data)
			if(len(my_string.split(','))==18 ):
				
				result = [[float(x.strip())] for x in (my_string.split(',')[2:-2])]
			
				eeg=np.append(eeg, result,axis = 1)
				
				if(send0.get()):
					events=np.array([[0,24]])
					print("send0")
					send0.set(0)
					events=np.append(events, [[len(eeg[0]),0]],axis=0)
				if(send1.get()):
					print("send1")
					send1.set(0)
					events=np.append(events, [[len(eeg[0]),1]],axis=0)
					threading.Thread(target=classify).start()
					
					
					
				
				if(send2.get()):
					events=np.array([[0,24]])
					print("send2")
					send2.set(0)
					events=np.append(events, [[len(eeg[0]),2]],axis=0)
				if(send3.get()):
					print("send3")
					send3.set(0)
					events=np.append(events, [[len(eeg[0]),3]],axis=0)
					threading.Thread(target=classify2,args=(0,)).start()
				
					
					
				if(send4.get()):
					events=np.array([[0,24]])
					print("send4")
					send4.set(0)
					events=np.append(events, [[len(eeg[0]),4]],axis=0)
				if(send5.get()):
					print("send5")
					send5.set(0)
					events=np.append(events, [[len(eeg[0]),5]],axis=0)
					threading.Thread(target=classify2,args=(1,)).start()

	print("finished")

	s.close()		
	