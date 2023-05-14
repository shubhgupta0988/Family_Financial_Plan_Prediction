import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import json
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
from flask_restful import Api, Resource, reqparse, abort, fields
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout

app = Flask(__name__)

dataset = pd.read_csv('inc_children_data_50.csv')

x = dataset.iloc[:, :1].values
y = dataset.iloc[:, 2].values

for i in range(0, len(x)):
  x[i, 0] = int(x[i, 0].replace(",", ""))
y = y.reshape(len(y), 1)

sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

regressor = SVR(kernel='rbf')
regressor.fit(x, y)

print("Training completed for Children .......")

x = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 0].values

for i in range(0, len(y)):
    y[i] = int(y[i].replace(",", ""))
y = y.reshape(len(y), 1)
sc_x_1 = StandardScaler()
sc_y_1 = StandardScaler()
x = sc_x_1.fit_transform(x)
y = sc_y_1.fit_transform(y)

regressor = SVR(kernel='rbf')
regressor.fit(x, y)

print("Training Completed for Income.......")


data = pd.read_csv('Google_train_data.csv')
data.head()

data["Close"]=pd.to_numeric(data.Close,errors='coerce')
data = data.dropna()
trainData = data.iloc[:,4:5].values

sc = MinMaxScaler(feature_range=(0,1))
trainData = sc.fit_transform(trainData)
trainData.shape

X_train = []
y_train = []

for i in range (60,1149): #60 : timestep // 1149 : length of the data
    X_train.append(trainData[i-60:i,0]) 
    y_train.append(trainData[i,0])

X_train,y_train = np.array(X_train),np.array(y_train)

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1)) #adding the batch_size axis
X_train.shape

model = Sequential()

model.add(LSTM(units=100, return_sequences = True, input_shape =(X_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = False))
model.add(Dropout(0.2))

model.add(Dense(units =1))
model.compile(optimizer='adam',loss="mean_squared_error")

hist = model.fit(X_train, y_train, epochs = 2, batch_size = 32, verbose=2)

#plt.plot(hist.history['loss'])
#plt.title('Training model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train'], loc='upper left')
#plt.show()

testData = pd.read_csv('Google_test_data.csv')
testData["Close"]=pd.to_numeric(testData.Close,errors='coerce')
testData = testData.dropna()
testData = testData.iloc[:,4:5]
y_test = testData.iloc[60:,0:].values 
#input array for the model
inputClosing = testData.iloc[:,0:].values 
inputClosing_scaled = sc.transform(inputClosing)
inputClosing_scaled.shape
X_test = []
length = len(testData)
timestep = 60
for i in range(timestep,length):  
    X_test.append(inputClosing_scaled[i-timestep:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
X_test.shape

y_pred = model.predict(X_test)

predicted_price = sc.inverse_transform(y_pred)
r = max(predicted_price) - min(predicted_price)
print(r)

#plt.plot(y_test, color = 'red', label = 'Actual Stock Price')
#plt.plot(predicted_price, color = 'green', label = 'Predicted Stock Price')
#plt.title('Google stock price prediction')
#plt.xlabel('Time')
#plt.ylabel('Stock Price')
#plt.legend()
#plt.show()

data1 = pd.read_csv('tesla.csv')
data1.head()

data1["Close"]=pd.to_numeric(data1.Close,errors='coerce')
data1 = data1.dropna()
trainData1 = data1.iloc[:,4:5].values

sc1 = MinMaxScaler(feature_range=(0,1))
trainData1 = sc1.fit_transform(trainData1)
trainData1.shape

X_train1 = []
y_train1 = []

for i in range (60,1149):
    X_train1.append(trainData1[i-60:i,0]) 
    y_train1.append(trainData1[i,0])

X_train1,y_train1 = np.array(X_train1),np.array(y_train1)

X_train1 = np.reshape(X_train1,(X_train1.shape[0],X_train1.shape[1],1))
X_train1.shape

model1 = Sequential()

model1.add(LSTM(units=100, return_sequences = True, input_shape =(X_train1.shape[1],1)))
model1.add(Dropout(0.2))

model1.add(LSTM(units=100, return_sequences = True))
model1.add(Dropout(0.2))

model1.add(LSTM(units=100, return_sequences = True))
model1.add(Dropout(0.2))

model1.add(LSTM(units=100, return_sequences = False))
model1.add(Dropout(0.2))

model1.add(Dense(units =1))
model1.compile(optimizer='adam',loss="mean_squared_error")

hist1 = model1.fit(X_train1, y_train1, epochs = 2, batch_size = 32, verbose=2)

testData1 = pd.read_csv('tesla_stock_test.csv')
testData1["Close"]=pd.to_numeric(testData1.Close,errors='coerce')
testData1 = testData1.dropna()
testData1 = testData1.iloc[:,4:5]
y_test1 = testData1.iloc[60:,0:].values 
inputClosing1 = testData1.iloc[:,0:].values 
inputClosing_scaled1 = sc1.transform(inputClosing1)
inputClosing_scaled1.shape
X_test1 = []
length1 = len(testData1)
timestep = 60
for i in range(timestep,length1):  
    X_test1.append(inputClosing_scaled1[i-timestep:i,0])
X_test1 = np.array(X_test1)
X_test1 = np.reshape(X_test1,(X_test1.shape[0],X_test1.shape[1],1))
X_test1.shape

y_pred1 = model.predict(X_test1)
predicted_price1 = sc1.inverse_transform(y_pred1)
r1 = max(predicted_price1) - min(predicted_price1)
print(r1)

if r1>r:
  print("Invest in Tesla stocks.")
else:
  print("Invest in Google stocks.")

@app.route('/predictChildren', methods=['POST'])
def predictChildren():
    content_type = request.headers.get('Content-Type')
    print("Content type : ", content_type)
    data = json.loads(request.data)

    print(data)
    print("income : ", data['income'])
    
    income = data['income']
    debt = data['debt']
    investment = data['investment']

    fin = income - debt + investment
    chi = math.floor(sc_y.inverse_transform(
        regressor.predict(sc_x.transform([[fin]])).reshape(-1, 1)))
    if chi > 3:
        chi = 3

    return jsonify({'Number of Children': str(chi)})

@app.route('/predictFamily', methods=['POST'])
def predictFamily():

    content_type = request.headers.get('Content-Type')
    print("Content type : ", content_type)
    data = json.loads(request.data)
    print(data)
    print("income : ", data['income'])

    income = data['income']

    dataset = pd.read_csv('fam_fin_plan.csv')
    p = dataset.iloc[:, 6:14].values

    for i in range(3143):
        for j in range(8):
            p[i][j] = p[i][j].replace("$", "")
            p[i][j] = int(p[i][j].replace(",", ""))

    for i in range(3143):
        for j in range(7):
            p[i][j] = (p[i][j]/p[i][7])*100

        """
        p_housing
        p_food
        p_transportation
        p_healthcare
        p_othernecessities
        p_childcare
        p_taxes
        """

    perc = [0]*7
    for i in range(7):
        for j in range(3143):
            perc[i] = perc[i] + p[j][i]
    for i in range(7):
        perc[i] = round(perc[i]/3143)/100
    print(perc)

    plan = [0]*7
    for i in range(7):
        # 60000*12 = 720000, is the cost for the parents which is removed(based on research)
        plan[i] = (income - 720000) * perc[i]

    return jsonify({"Housing": plan[0], "Food": plan[1], "Transportation": plan[2], "Healthcare": plan[3], "Other Necessities": plan[4], "Childcare": plan[5], "Taxes": plan[6]})    

@app.route('/predictIncome', methods=['POST'])
def predictIncome():

    content_type = request.headers.get('Content-Type')
    print("Content type : ", content_type)
    data = json.loads(request.data)
    print(data)
    print("childrens : ", data['childrens'])

    n = data['childrens']

    t = sc_y_1.inverse_transform(regressor.predict(
        sc_x_1.transform([[n]])).reshape(-1, 1))
    tval = t[0][0]

    return jsonify({"Income": tval})

@app.route('/predictStock', methods=['POST'])
def predictStock():
    content_type = request.headers.get('Content-Type')
    print("Content type : ", content_type)
    data = json.loads(request.data)

    if r1>r:
      return jsonify({'Stock': 'Tesla'})
    else:
      return jsonify({'Stock': 'Google'})


if __name__ == "__main__":
    app.run(debug=True)
