import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
df=pd.read_csv('deneme.csv')
veri=df.copy()
veri.drop(['Date','Volume','Open','High','Low'],1,inplace=True)
predictionDays = 30
veri['Prediction'] = veri[['Close/Last']].shift(-predictionDays)
x=np.array(veri.drop(['Prediction'],1))
x = x[:len(veri)-predictionDays]
y=np.array(veri['Prediction'])
y = y[:-predictionDays]
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.2)
predictionDays_array = np.array(veri.drop(['Prediction'],1))[-predictionDays:]
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001)
svr_rbf.fit(xtrain, ytrain)
svr_rbf_confidence = svr_rbf.score(xtest,ytest)
print('SVR_RBF:',svr_rbf_confidence)
print('======================================================')
svm_prediction = svr_rbf.predict(xtest)
print(svm_prediction)
print(ytest)
print('======================================================')
svm_prediction = svr_rbf.predict(predictionDays_array)
print(svm_prediction)
print(veri.tail(predictionDays))
print('======================================================')
