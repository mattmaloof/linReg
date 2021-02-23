import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv(
    r'C:\Users\Matthew\Desktop\Machine Learning\data1.txt', header=None) # No header so the first row of data is not skipped.

x = data.iloc[:, 0]   # Reading first column of data file assigning the values to x.
y = data.iloc[:, 1]
x = np.array(x)       # Turning x and turning it into an array.
y = np.array(y)

denom = x.dot(x) - (x.mean() * x.sum())                 # The denominator for linear regression is the dot product of x and x 
                                                        # minus the mean of x multiplied by the summation of x.

m = (x.dot(y) - y.mean() * x.sum()) / denom             # m = slope (Gradient)
b = (y.mean() * x.dot(x) - x.mean() * x.dot(y)) / denom # b = y-intercept (Bias)

plt.title("Matthew Maloof", fontsize=(20))
plt.scatter(x,y)
plt.plot(x, m*x+b, color="red", label="Line of Best Fit") # y=mx+b is the formula for linear regression.
plt.legend(loc="upper left")

model = LinearRegression()
model.fit(x.reshape(-1, 1),y.reshape(-1, 1))
score = model.score(x.reshape(-1, 1), y.reshape(-1, 1))

data = pd.read_csv(
    r'C:\Users\Matthew\Desktop\Machine Learning\data2.txt', header=None) # No header so the first row of data is not skipped.

xy = data.iloc[:, 0:2]   # Reading first two columns of data file assigning the values to x.
z = data.iloc[:, 2]

model = LinearRegression()
model.fit(xy,z)

model.coef_             # Calculates the slope coefficients for x and y (both independent variables).
model.intercept_        # Calculates the intercept for the model.
model.score(xy,z)       # Calculates the accuracy score for the model.