#!/usr/bin/env python
# coding: utf-8

# Jupyter notebook 
# 1st application in python

# Import pandas, sklearn, matplotlib

import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import sys

print(sys.argv[1])


print("Running linear modeling of data python script")
print()


# Set notebook variables

filename = sys.argv[1]

base,ext = os.path.splitext(filename)
print(base)
print(ext)


filename = "regrex1.csv"
print("Loading filename {}".format(filename))
print()



# Use the read_csv() function to read regrex1.csv file


dataset = pd.read_csv(filename)
print(dataset)


dataset.describe()


# Fitting Linear Regression to the Dataset


model = LinearRegression()
model.fit(dataset[['x']], dataset[['y']])


# Adjusted R-squared

model.score(dataset[['x']], dataset[['y']])


# Visualizing the Linear Regression results


plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.title("Linear model of y vs x for {}".format(filename))
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.savefig("{}.png".format(base)) 


plt.plot(dataset[['x']], model.predict(dataset[['x']]), color = 'blue')
plt.title("Linear model of y vs x for {}".format(filename))
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.savefig("{}_model.png".format(base))


