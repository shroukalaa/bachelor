from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import pickle
#file = open("shaker.txt","a") 
#11.9,8.06,6.41,4.54	#23.8,16.12,12.82,9.08 	
Fs=128.0
N = 641;
T = 1.0 / Fs
f = np.linspace(0.0, 1.0/(2.0*T), N//2)
window=0.2

#c4=[any(y) for y in zip ([all(x) for x in zip(f<23.8+window, f>23.8-window)],[all(x) for x in zip(f<23.8+window, f>23.8-window)])]
#c3=[any(y) for y in zip ([all(x) for x in zip(f<16.12+window, f>16.12-window)],[all(x) for x in zip(f<12.82+window, f>12.82-window)])]
#c2=[any(y) for y in zip ([all(x) for x in zip(f<12.82+window, f>12.82-window)],[all(x) for x in zip(f<12.82+window, f>12.82-window)])]
#c1=[any(y) for y in zip ([all(x) for x in zip(f<9.08+window, f>9.08-window)],[all(x) for x in zip(f<9.08+window, f>9.08-window)])]

c1=[any(y) for y in zip ([all(x) for x in zip(f<12+window, f>12-window)],[all(x) for x in zip(f<24+window, f>24+window)])]
c2=[any(y) for y in zip ([all(x) for x in zip(f<10+window, f>10-window)],[all(x) for x in zip(f<20+window, f>20+window)])]
c3=[any(y) for y in zip ([all(x) for x in zip(f<7.5+window, f>7.5-window)],[all(x) for x in zip(f<15+window, f>15+window)])]
c4= [any(y) for y in zip ([all(x) for x in zip(f<6.66+window, f>6.66-window)],[all(x) for x in zip(f<13.32+window, f>13.32+window)])]


matr2 = np.array([[0.00, 12.00 , 10.00, 7.50,6.66  ], \
					 [12.0, 0.00, 0.00, 0.00, 0.00], \
					 [10.0, 0.00, 0.00, 0.00, 0.00], \
					 [7.50, 0.00, 0.00, 0.00, 0.00], \
					 [6.66,0.00, 0.00, 0.00, 0.00]])

labels=[1,2,3,4]
data_labels=[ 1,2,4,3,4,2,1,3,1,2,3,4,4,3,2,1,3,1,2,4,1,3,4,2, 
1,2,4,3,4,2,1,3,1,2,3,4,4,3,2,1,3,1,2,4,1,3,4,2,1,2,4,3,4,2,1,3,1,2,3,4,4,3,2,1,3,1,2,4,1,3,4,2, 
1,2,4,3,4,2,1,3,1,2,3,4,4,3,2,1,3,1,2,4,1,3,4,2,1,2,4,3,4,2,1,3,1,2,3,4,4,3,2,1,3,1,2,4,1,3,4,2]       
       
data_labels2=np.array([ 1,2,4,3,4,2,1,3,1,2,3,4,4,3,2,1,3,1,2,4,1,3,4,2  ,
1,2,4,3,4,2,1,3,1,2,3,4,4,3,2,1,3,1,2,4,1,3,4,2,1,2,4,3,4,2,1,3,1,2,3,4,4,3,2,1,3,1,2,4,1,3,4,2, 
1,2,4,3,4,2,1,3,1,2,3,4,4,3,2,1,3,1,2,4,1,3,4,2,1,2,4,3,4,2,1,3,1,2,3,4,4,3,2,1,3,1,2,4,1,3,4,2])
dataindex=0;
#data=np.zeros( (120, 147) )

data=np.zeros( (120,320 ) )
maximum=[]
subject="mostafa"
#------------------------------------------------------------------------
x = loadmat(subject+'1.mat')

events=x['events']
codes=events[:,0:2]
eeg = x['eeg'][0:14,:]
mean= np.mean(eeg, axis=0)

condition1 = codes[:,1]==0
condition2 = codes[:,1]==1
condition= [any(x) for x in zip(condition1, condition2)]
code=np.extract(condition, codes[:,0])

#print ("...")
#print (code)
#print ("...")
for i in range(0,len(code),2):

		before=eeg[6,code[i]-1:code[i]+640]
		after=before - mean[code[i]-1:code[i]+640]

		y = fft(after)
		p1=2.0/N * np.abs(y[0:N//2])
		
		##data[dataindex]=p1[35:182]
		data[dataindex]=p1
		dataindex=dataindex+1
		maxvalues= [] 
		#print(np.sum(np.extract(c1, p1)))
		#print(np.sum(np.extract(c2, p1)))
		#print(np.sum(np.extract(c3, p1)))
		#print(np.sum(np.extract(c4, p1)))
		#print("------------")
		maxvalues.append(max(np.extract(c1, p1)))
		maxvalues.append(max(np.extract(c2, p1)))
		maxvalues.append(max(np.extract(c3, p1)))
		maxvalues.append(max(np.extract(c4, p1)))
		j=labels.index(data_labels[i//2])
		l=maxvalues.index(max(maxvalues))
		matr2[l+1,j+1]=matr2[l+1,j+1]+1


		
#------------------------------------------------------------------------
x = loadmat(subject+'2.mat')

events=x['events']
codes=events[:,0:2]
eeg = x['eeg'][0:14,:]
mean= np.mean(eeg, axis=0)

condition1 = codes[:,1]==0
condition2 = codes[:,1]==1
condition= [any(x) for x in zip(condition1, condition2)]
code=np.extract(condition, codes[:,0])
#print(code)
for i in range(0,len(code),2):

		before=eeg[6,code[i]-1:code[i]+640]
		after=before - mean[code[i]-1:code[i]+640]

		y = fft(after)
		p1=2.0/N * np.abs(y[0:N//2])
		
		#data[dataindex]=p1[35:182]
		data[dataindex]=p1
		dataindex=dataindex+1
		maxvalues= [] 
		maxvalues.append(max(np.extract(c1, p1)))
		maxvalues.append(max(np.extract(c2, p1)))
		maxvalues.append(max(np.extract(c3, p1)))
		maxvalues.append(max(np.extract(c4, p1)))
		j=labels.index(data_labels[i//2])
		l=maxvalues.index(max(maxvalues))
		matr2[l+1,j+1]=matr2[l+1,j+1]+1
		
#------------------------------------------------------------------------
x = loadmat(subject+'3.mat')

events=x['events']
codes=events[:,0:2]
eeg = x['eeg'][0:14,:]
mean= np.mean(eeg, axis=0)

condition1 = codes[:,1]==0
condition2 = codes[:,1]==1
condition= [any(x) for x in zip(condition1, condition2)]
code=np.extract(condition, codes[:,0])
#print(code)
for i in range(0,len(code),2):

		before=eeg[6,code[i]-1:code[i]+640]
		after=before - mean[code[i]-1:code[i]+640]

		y = fft(after)
		p1=2.0/N * np.abs(y[0:N//2])
		
		#data[dataindex]=p1[35:182]
		data[dataindex]=p1
		dataindex=dataindex+1
		maxvalues= [] 
		maxvalues.append(max(np.extract(c1, p1)))
		maxvalues.append(max(np.extract(c2, p1)))
		maxvalues.append(max(np.extract(c3, p1)))
		maxvalues.append(max(np.extract(c4, p1)))
		j=labels.index(data_labels[i//2])
		l=maxvalues.index(max(maxvalues))
		matr2[l+1,j+1]=matr2[l+1,j+1]+1
		
		
#------------------------------------------------------------------------
x = loadmat(subject+'4.mat')

events=x['events']
codes=events[:,0:2]
eeg = x['eeg'][0:14,:]
mean= np.mean(eeg, axis=0)

condition1 = codes[:,1]==0
condition2 = codes[:,1]==1
condition= [any(x) for x in zip(condition1, condition2)]
code=np.extract(condition, codes[:,0])
#print(code)
for i in range(0,len(code),2):

		before=eeg[6,code[i]-1:code[i]+640]
		after=before - mean[code[i]-1:code[i]+640]

		y = fft(after)
		p1=2.0/N * np.abs(y[0:N//2])
		
		#data[dataindex]=p1[35:182]
		data[dataindex]=p1
		dataindex=dataindex+1
		maxvalues= [] 
		maxvalues.append(max(np.extract(c1, p1)))
		maxvalues.append(max(np.extract(c2, p1)))
		maxvalues.append(max(np.extract(c3, p1)))
		maxvalues.append(max(np.extract(c4, p1)))
		j=labels.index(data_labels[i//2])
		l=maxvalues.index(max(maxvalues))
		matr2[l+1,j+1]=matr2[l+1,j+1]+1
		
#------------------------------------------------------------------------
x = loadmat(subject+'5.mat')

events=x['events']
codes=events[:,0:2]
eeg = x['eeg'][0:14,:]
mean= np.mean(eeg, axis=0)

condition1 = codes[:,1]==0
condition2 = codes[:,1]==1
condition= [any(x) for x in zip(condition1, condition2)]
code=np.extract(condition, codes[:,0])

#print(code)
for i in range(0,len(code),2):

		before=eeg[6,code[i]-1:code[i]+640]
		after=before - mean[code[i]-1:code[i]+640]

		y = fft(after)
		p1=2.0/N * np.abs(y[0:N//2])
		data[dataindex]=p1
		#data[dataindex]=p1[35:182]
		##data[dataindex]=p1[35:182][21:181]

		
		dataindex=dataindex+1
		maxvalues= [] 
		maxvalues.append(max(np.extract(c1, p1)))
		maxvalues.append(max(np.extract(c2, p1)))
		maxvalues.append(max(np.extract(c3, p1)))
		maxvalues.append(max(np.extract(c4, p1)))
		j=labels.index(data_labels[i//2])
		l=maxvalues.index(max(maxvalues))
		matr2[l+1,j+1]=matr2[l+1,j+1]+1



#import seaborn as sns
#import seaborn as sns

#import matplotlib.pyplot as plt
#import pandas


#data1=pd.DataFrame(data=data[:,:])

#corr = data1.corr()
#corr.to_csv(r'shaker.csv')
#f, ax = plt.subplots(figsize=(10, 10))
#cmap = sns.diverging_palette(220, 10, as_cmap=True)
#sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5)
#plt.show()
		

# data for tow clsses
condition1 = data_labels2==1
condition2 = data_labels2==3
#print(condition1)
#print(condition2)
condition= [any(x) for x in zip(condition1, condition2)]
data_labels2=np.extract(condition, data_labels[:])
#print(data_labels2)
data2=np.zeros((60,320));

n=0
for k in range(0,len(condition),1):
	if(condition[k]==1):
		data2[n]=data[n]
		n+=1

#print("labels lenght =",len(data_labels2))
#print("data lenght =",len(data2))
#print("data0 lenght =",len(data2[0]))

#data=data[0:168]
#data_labels=data_labels[0:168]
#data2=data2[0:84]
#data_labels2=data_labels2[0:84]
print("train")	
X_train, X_test, y_train, y_test = train_test_split(data, data_labels, test_size = 0.2,shuffle=True, random_state = 0)

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, random_state = 0)
rf.fit(X_train, y_train)

filename = 'finalized_model_mostafa6.sav'
pickle.dump(rf, open(filename, 'wb'))

y_pred = rf.predict(X_test)
y_proba=rf.predict_proba(X_test)

print(y_test)
print(y_pred)
print (accuracy_score(y_test, y_pred))

X_train, X_test, y_train, y_test = train_test_split(data, data_labels, test_size = 0.2,shuffle=True, random_state = 0)
sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  

X_test = sc.transform(X_test) 

filename = 'finalized_model_mostafa6sc.sav'
pickle.dump(sc, open(filename, 'wb'))





from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear',random_state = 0)  
svclassifier.fit(X_train, y_train)  

filename = 'finalized_model_mostafa6svm.sav'
pickle.dump(svclassifier, open(filename, 'wb'))

y_pred = svclassifier.predict(X_test) 
print(y_test)
print(y_pred)
print (accuracy_score(y_test, y_pred))

# train for 2 freq
X_train, X_test, y_train, y_test = train_test_split(data2, data_labels2, test_size = 0.2,shuffle=True, random_state = 0)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, random_state = 0)
rf.fit(X_train, y_train)



y_pred = rf.predict(X_test)
y_proba=rf.predict_proba(X_test)


print(y_test)
print(y_pred)
print (accuracy_score(y_test, y_pred))

#filename = 'finalized_model_es6_2_rf.sav'
#pickle.dump(rf , open(filename, 'wb'))


X_train, X_test, y_train, y_test = train_test_split(data2, data_labels2, test_size = 0.2,shuffle=True, random_state = 0)
sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  

X_test = sc.transform(X_test) 

#filename = 'finalized_model_es6_2_sc.sav'
#pickle.dump(sc, open(filename, 'wb'))




from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear',random_state = 0)  
svclassifier.fit(X_train, y_train)  

#filename = 'finalized_model_es6_2.sav'
#pickle.dump(svclassifier, open(filename, 'wb'))


y_pred = svclassifier.predict(X_test) 
print(y_test)
print(y_pred)
print (accuracy_score(y_test, y_pred))


			
for  row in  matr2:
	print ('[%s]' % (' '.join('%06s' % i for i in row)))
	#file.write('[%s]' % (' '.join('%06s' % i for i in row))+'\n')