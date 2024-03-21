# Importing core libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Importing dataset
data = pd.read_csv("./data.csv")

# X -> Independent variable
# Y -> dependent variable
X = data.iloc[:, 0]
y = data.iloc[:, -1]

# Splitting data into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Creating insance for regression model
regressor = LinearRegression()

# Fitting the data to regressor instance
# Note: we need to reshape the 1D array to 2D array
regressor.fit(np.array(X_train).reshape(-1, 1), np.array(y_train).reshape(-1, 1))

# Predicting the results using trained model
y_pred = regressor.predict(np.array(X_test).reshape(-1, 1))

# Plotting the final results using matplotlib plottijng library
plt.scatter(X_test, y_test, color="red")
plt.plot(X_test, y_pred, color="blue")
plt.show()