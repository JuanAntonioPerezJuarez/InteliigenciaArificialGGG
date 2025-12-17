# Simple Linear Regression Example, Juan Antonio Pérez Juárez
# This script demonstrates a simple linear regression using Python's sklearn library.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data
x = np.array([2, 4, 6, 8, 10]).reshape(-1, 1)  # Hours studied (reshaped for sklearn)
y = np.array([81, 93, 91, 97, 99])             # Exam scores

# Create and fit the model
model = LinearRegression()
model.fit(x, y)

# Predict values for plotting the regression line
x_pred = np.linspace(2, 10, 100).reshape(-1, 1)
y_pred = model.predict(x_pred)

# Plot the data points and regression line
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x_pred, y_pred, color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Simple Linear Regression Example')
plt.legend()
plt.grid(True)
plt.show()


# Print the coefficients
print(f"Intercept (β₀): {model.intercept_:.2f}")
print(f"Slope (β₁): {model.coef_[0]:.2f}")

# Predict an exam score for a student who studied 7 hours
hours = 7
predicted_score = model.predict([[hours]])
print(f"Predicted exam score for {hours} hours studied: {predicted_score[0]:.2f}")
