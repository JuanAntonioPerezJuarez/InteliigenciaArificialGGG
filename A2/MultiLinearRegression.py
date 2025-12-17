### Multiple Linear Regression Example, Juan Antonio Pérez Juárez
# This script demonstrates a multiple linear regression using Python's sklearn library
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Data
X = np.array([
    [2, 7],
    [4, 6],
    [6, 8],
    [8, 8],
    [10, 9]
])
y = np.array([81, 89, 92, 95, 99])

# Fit the model
model = LinearRegression()
model.fit(X, y)

# Coefficients
print(f"Intercept (β₀): {model.intercept_:.2f}")
print(f"Coefficient for Hours Studied (β₁): {model.coef_[0]:.2f}")
print(f"Coefficient for Hours Slept (β₂): {model.coef_[1]:.2f}")

# Predict an exam score for a student who studied 7 hours and slept 7 hours
predicted_score = model.predict([[7, 7]])
print(f"Predicted exam score for 7 hours studied and 7 hours slept: {predicted_score[0]:.2f}")

# 3D Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], y, color='blue', label='Data Points')

# Plot regression plane
x1_range = np.linspace(2, 10, 10)
x2_range = np.linspace(6, 9, 10)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
y_pred_grid = (model.intercept_ + 
               model.coef_[0] * x1_grid + 
               model.coef_[1] * x2_grid)
ax.plot_surface(x1_grid, x2_grid, y_pred_grid, alpha=0.5, color='red')

ax.set_xlabel('Hours Studied')
ax.set_ylabel('Hours Slept')
ax.set_zlabel('Exam Score')
ax.set_title('Multiple Linear Regression Example')
plt.legend()
plt.show()