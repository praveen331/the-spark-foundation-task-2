# the-spark-foundation-task-2
In [ ]:
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
In [ ]:
#Reading data from the PDF
url="http://bit.ly/w-data"
data=pd.read_csv(url)
In [ ]:
#viewing 10 records from the top of the file
data.head(10)
Out[ ]:
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
5	1.5	20
6	9.2	88
7	5.5	60
8	8.3	81
9	2.7	25
In [ ]:
data.plot(x='Hours',y='Scores',style='.')
plt.title('Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Score percentage')
plt.grid()
axis=plt.gca()
axis.set_facecolor('#FFE873')
plt.show()

In [ ]:
#Dividing data into input and output
x=data.iloc[:,].values
y=data.iloc[:,1].values
In [ ]:
#Dividing data into training data and testing data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)
In [ ]:
#Training our model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(xtrain,ytrain)
Out[ ]:
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
In [ ]:
line=regressor.coef_*x+regressor.intercept_
plt.scatter(x,y,marker='x')
plt.plot(x,line,'--r')
plt.grid()
axis=plt.gca()
axis.set_facecolor('#FFE873')
plt.show()

In [ ]:
#Predicting values of test data
ypred=regressor.predict(xtest)
df=pd.DataFrame({'Actual':ytest,'Predicted':ypred})
df
Out[ ]:
Actual	Predicted
0	20	16.884145
1	27	33.732261
2	69	75.357018
3	30	26.794801
4	62	60.491033
In [ ]:
df.plot.bar(figsize=(5,4))
Out[ ]:
<matplotlib.axes._subplots.AxesSubplot at 0x7fb21ede1668>

In [ ]:
#Predicting Score for 9.25 hours of study
hour=[[9.25]] #regressor.predict expects 2D array as the provided input
ypredict=regressor.predict(hour)
print("Hours",9.25)
print("Score",ypredict[0])
Hours 9.25
Score 93.69173248737539
In [ ]:
#model evaluation
from sklearn import metrics
print("Mean Absolute Error",metrics.mean_absolute_error(ytest,ypred))
Mean Absolute Error 4.183859899002982
