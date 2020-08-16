#This code illustrates polynomial regression of order 3, that is y = x^3 + x^2 + x + b
#to find the best fit line through the data.


#import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Training set
x_train = [[0], [5], [10], [15], [20], [25], [30], [35], [40], [50]] #Temperature of water in a tank
y_train = [[1], [49], [75], [80], [78],[83], [100], [150], [200], [400]] #pressure as the temperature increases

# Testing set
x_test = [[3], [6], [9], [12], [21], [27], [33], [36], [42], [51]] #Temperature of water in a tank
y_test = [[15], [55], [70], [69], [82], [100], [110], [150], [220], [450]] #pressure as the temperature increases


# Train the Linear Regression model and plot a prediction
regressor = LinearRegression()
regressor.fit(x_train, y_train)
xx = np.linspace(0, 60, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)


# Set the degree of the Polynomial Regression model
cubic_featurizer = PolynomialFeatures(degree=3)

# transforming input data matrix into a new data matrix of the given degree
x_train_cubic = cubic_featurizer.fit_transform(x_train)
x_test_cubic = cubic_featurizer.transform(x_test)

# Train and test the regressor_quadratic model
regressor_cubic = LinearRegression()
regressor_cubic.fit(x_train_cubic, y_train)
xx_cubic = cubic_featurizer.transform(xx.reshape(xx.shape[0], 1))

# Plot the graph
plt.plot(xx, regressor_cubic.predict(xx_cubic), c='r', linestyle='-.')
plt.title('Pressure of tank regressed on water Temperature')
plt.xlabel('Temperature(degrees c)')
plt.ylabel('Pressure (bar)')
plt.legend(['Linear Regression model', 'Polynomial Regression model'])
#plt.legend('Polynomial Regression model')
plt.grid(True)
plt.scatter(x_train, y_train)
plt.show()
print (x_train)
print (x_train_cubic)
print (x_test)
print (x_test_cubic)
