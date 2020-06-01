import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder



# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)



#print(data)




data=['C11'
,'B3'
,'B2'
,'A7'
,'A1'
,'D4'
,'E18'
,'D14'
,'C6'
,'E5'
,'A15'
,'E13']

dis=data
values = array(data)
values =sorted(values )

print(values)
# integer encode

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

print(onehot_encoded)

temp=[]
# onehot_encoded=np.concatenate((onehot_encoded[1], [values[1]]))

temp=np.concatenate((onehot_encoded[0], [values[0]]))
for i in range(1,len(values)):
    print(values[i])
    temp=np.vstack((temp,(np.concatenate((onehot_encoded[i], [values[i]])))))

print(temp)
onehot_encoded=temp

#onehot_encoded=pd.concat(onehot_encoded,values)


districts = pd.DataFrame(onehot_encoded, columns=[*values,"District"])
print(districts)

# print(values)
# dat = [[onehot_encoded, values]]
# # print(dat)
# # temp = pd.DataFrame(dat, columns=[*values,'nblank'])
# # print(temp)
# print(pd.concat(districts,values))

data=pd.read_csv('train_file.CSV')
print(data)
train_result=data.set_index('District').join(districts.set_index('District'))
# result = pd.([districts, data], axis=1).reindex(districts.index)

data=pd.read_csv('test_file.CSV')
print(data)
test_result=data.set_index('District').join(districts.set_index('District'))
# result = pd.([districts, data], axis=1).reindex(districts.index)


print(test_result)
print(train_result)

X_train_2 = train_result.drop(['num_crime',*dis], axis=1)
X_train = train_result.drop(['num_crime'], axis=1)
X_train_base = train_result[values]

print()
print(X_train_base)

y_train = train_result['num_crime']

X_test_2 = test_result.drop(['num_crime',*dis],axis=1)
X_test = test_result.drop(['num_crime'],axis=1)
X_test_base = test_result[values]

y_test = test_result['num_crime']


print(X_test)

regressor_2 = LinearRegression()
regressor = LinearRegression()
base = LinearRegression()
regressor_2.fit(X_train_2, y_train)
regressor.fit(X_train, y_train) #training the algorithm
base.fit(X_train_base, y_train)

#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(X_train_2.columns)
print(regressor_2.coef_)

y_pred_2 = regressor_2.predict(X_test_2)
y_pred = regressor.predict(X_test)
y_pred_base = base.predict(X_test_base)

print(y_pred)
print(y_pred_base)
print(y_test)

pred_2_error = abs(y_pred_2 - y_test)
pred_error = abs(y_pred - y_test)
base_error = abs(y_pred_base - y_test)
print(y_pred - y_test)
print(y_pred_base - y_test)

print('Average Prediction Error 2: ', np.average(pred_2_error))
print('Average Prediction Error: ', np.average(pred_error))
print('Average Base Error: ', np.average(base_error))

plt.plot(y_train.keys(), y_test, 'o', label='Actual')
plt.plot(y_train.keys(), y_pred_2, 'o', label='Prediction Rips')
plt.plot(y_train.keys(), y_pred, 'o', label='Prediction Rips and District')
plt.plot(y_train.keys(), y_pred_base, 'o', label='Prediction Base')

plt.ylabel('number of crimes')
plt.xlabel('District')
plt.legend()
plt.show()

x = np.arange(len(y_train.keys()))
width = 0.35

fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, pred_error, width, label='Rips and District Based Error')
rects2 = ax.bar(x + width/2, base_error, width, label='District Based Error')
rects3 = ax.bar(x + width/2,pred_2_error, width, label='Rips Based Error')

ax.set_ylabel('Error')
#ax.set_xticks(x)
#ax.set_xticklabels(y_train.keys())
ax.legend()

fig.tight_layout()
plt.show()