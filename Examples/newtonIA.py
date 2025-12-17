import numpy as np
import matplotlib.pyplot as plt

#Definicion de la funcion

def f(x, y):
    return 0.5*(3*x**2 + 2*x*y + 2*y**2) - 5*x - 3*y

#Optimo Analitico (H=[[3,1],[1,2]], b=[5,3] -> w* = H^{-1}*b = [1.4,0.8])
w_star = np.array([1.4, 0.8])
z_star = f(*w_star)

#Malla para evaluar f
x_min, x_max = -2, 4
y_min, y_max = -2, 4

x = np.linspace(x_min, x_max, 100)
y = np.linspace(y_min, y_max, 100)  
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

#Gradiente y Hessiano
def grad_f(w):  
    x, y = w
    dfdx = 3*x + y - 5
    dfdy = x + 2*y - 3
    return np.array([dfdx, dfdy])

def hess_f(w):
    return np.array([[3, 1], [1, 2]])

#Metodo de Newton
def newton_method(w0, tol=1e-6, max_iter=100):  
    w = w0
    path = [w0]
    for i in range(max_iter):
        grad = grad_f(w)
        hess = hess_f(w)
        if np.linalg.norm(grad) < tol:
            break
        w = w - np.linalg.inv(hess).dot(grad)
        path.append(w)
    return np.array(path)

#Punto inicial
w0 = np.array([0.0, 0.0])
path = newton_method(w0)
z_path = f(path[:, 0], path[:, 1])
num_iters = path.shape[0]

#Grafica de contornos y trayectoria
plt.figure(figsize=(10, 6)) 
contours = plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)
plt.plot(path[:, 0], path[:, 1], marker='o', color='red', label='Trayectoria de Newton')
plt.scatter(w_star[0], w_star[1], color='blue', label='Óptimo Analítico')
plt.title('Método de Newton')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#Grafica de la funcion a lo largo de la trayectoria
plt.figure(figsize=(10, 6)) 
plt.plot(range(num_iters), z_path, marker='o')
plt.axhline(y=z_star, color='r', linestyle='--', label='Óptimo Analítico')
plt.title('Valor de la función a lo largo de la trayectoria')
plt.xlabel('Iteraciones')
plt.ylabel('Valor de la función')
plt.legend()
plt.show()

#Resultados
print(f'Número de iteraciones: {num_iters}')
print(f'Punto encontrado: {path[-1]}')
print(f'Valor de la función en el punto encontrado: {z_path[-1]}')
