import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression




data = pd.read_csv("..\datasets\studentscores.csv")
# print(data.head())
# print(data.shape)

X= data.iloc[: ,:1].values
Y= data.iloc[: ,1].values

x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.2)

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)


reg = LinearRegression()
reg = reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

plt.scatter(x_train , y_train , color='red')
plt.plot(x_train , reg.predict(x_train), color='blue')
plt.show()


plt.scatter(x_test, y_test , color='blue')
plt.plot(x_test , reg.predict(x_test), color='red')

plt.show()