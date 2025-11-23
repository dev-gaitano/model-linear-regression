from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression
import numpy as np

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20,
                                random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=1234)

# Modelling
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = np.mean((y_test - y_pred) ** 2)
print("MSE       : %.2f" % mse)

# Calculate R-squared
y_test = np.array(y_test)
y_mean = y_test.mean()
ss_total = np.sum((y_test - y_mean) ** 2)
ss_res = np.sum((y_test - y_pred) ** 2)
r_squared = 1 - (ss_res / ss_total)
r_squared_percent = r_squared * 100

print("Y MEAN    : %.2f" % y_mean)
print("R-squared : %.2f%%" % r_squared_percent)

# Plot Data
y_pred_line = model.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()
