import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('../datasets/data.csv')

print(df.head())


# X and Y 
X = df.iloc[: , :-1].values
Y = df.iloc[: , -1].values



imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[: , 1:3])
X[: , 1:3] = imputer.transform(X[ : , 1:3])





labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])


ct = ColumnTransformer([('Country' ,OneHotEncoder() , [0] )] , remainder='passthrough')

X = ct.fit_transform(X)
print(X)

labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)    

x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size=0.2)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

sc_c = StandardScaler()

x_train[: , 3:] = sc_c.fit_transform(x_train[: , 3:])
x_test[:, 3:] = sc_c.fit_transform(x_test[:, 3:])
print(x_train)
