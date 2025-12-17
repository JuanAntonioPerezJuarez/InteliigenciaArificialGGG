# Logistic Regression for Admission Prediction, Juan Antonio Pérez Juárez
# This script demonstrates logistic regression using Python's sklearn library.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Data
X = np.array([
    [45, 85],
    [50, 43],
    [62, 70],
    [75, 80],
    [80, 90],
    [52, 65],
    [60, 60],
    [47, 56],
    [90, 88],
    [85, 72]
])
y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])

# Fit the model
model = LogisticRegression()
model.fit(X, y)

# Visualize data points
plt.figure(figsize=(8,6))
for label, marker, color in zip([0,1], ['o', 's'], ['red', 'green']):
    plt.scatter(X[y==label, 0], X[y==label, 1], marker=marker, color=color, label=f'Admitted={label}', s=100)

plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.title('Logistic Regression: Admission Prediction')

# Plot decision boundary
x_min, x_max = X[:,0].min()-5, X[:,0].max()+5
y_min, y_max = X[:,1].min()-5, X[:,1].max()+5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)
plt.contour(xx, yy, probs, levels=[0.5], linewidths=2, colors='blue')

plt.legend()
plt.grid(True)
plt.show()

# Predict admission probability for a student with Exam 1=65, Exam 2=75
score = np.array([[65, 75]])
prob_admit = model.predict_proba(score)[0][1]
print(f"Predicted probability of admission for Exam 1=65, Exam 2=75: {prob_admit:.2f}")
