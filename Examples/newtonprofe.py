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

#Contornos 2d 
plt.figure(figsize=(6.5, 6))
cs = plt.contour(X, Y, Z, levels=25)
plt.clabel(cs, inline=True, fontsize=8)
plt.scatter(w_star[0], w_star[1], s=60) # Punto óptimo
plt.annotate("w* = 1.4, 0.8", (w_star[0], w_star[1]), textcoords="offset points", xytext=(5,5))
plt.title('Contornos de f(x,y)')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(True)
plt.savefig("Contornos_f_xy.png", bbox_inches='tight', dpi=200)

#Superficie 3d
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=0, antialiased=True)
ax.scatter([w_star[0]], [w_star[1]], [z_star], s= 50) # Punto óptimo
ax.text(w_star[0], w_star[1], z_star, "w*", zdir=None)
ax.set_title('Superficie de f(x,y)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x,y)')
plt.savefig("Superficie_f_xy.png", bbox_inches='tight', dpi=200)
plt.show()