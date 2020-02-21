import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Read data
train = pd.read_csv('train.csv')

#Display the scatter plot of GarageArea and SalePrice
plt.scatter(train.GarageArea, train.SalePrice, color='red')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.show()

# calculate interquartile range
print(train.GarageArea.describe())

#Delete the outlier value of GarageArea
outlier_drop = train[(train.GarageArea >334) & (train.GarageArea <576)]

##Display the scatter plot of GarageArea and SalePrice after deleting
plt.scatter(outlier_drop.GarageArea, outlier_drop.SalePrice, color='green')
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.show()