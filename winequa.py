import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

wine_quality = pd.read_csv('winequality-red.csv')

##handling missing value

dataset = wine_quality.select_dtypes(include=[np.number]).interpolate().dropna()

numeric_features  = dataset.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print(corr['quality'].sort_values(ascending=False)[1:4],'\n')
print(corr['quality'].sort_values(ascending=False)[-3:])

##Build a linear model

X=dataset.loc[:,['sulphates','volatile acidity','total sulfur dioxide','density','alcohol','citric acid']]
y=np.log(dataset.iloc[:,11].values)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
model = regressor.fit(X_train, y_train)

##Evaluate the performance and visualize results

print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)

from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))