import numpy as np
import matplotlib.pyplot as plt

# Funcion de Costo Simple
f = lambda x: x**2
grad = lambda x: 2*x

#Descenso de Gradiente
x = 10
eta = 0.1
hist = [x]

for i in range(20):
    x = x - eta * grad(x)
    hist.append(x)

plt.plot(range(len(hist)), [f(h) for h in hist], marker='o')
plt.xlabel('Iteraciones')
plt.ylabel('Valor de la Funcion de Costo')
plt.title('Descenso de Gradiente')
plt.show()