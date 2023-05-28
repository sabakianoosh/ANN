import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as met
import sklearn.linear_model as li


np.random.seed(0)


nD = 200
nX = 3
X = np.zeros((nD,nX))
Y = np.zeros((nD,1))


for i in range(nD):
    noise  = np.random.uniform(-1,1)
    X [i, :] = np.random.uniform(-1,1,nX)
    Y [i , 0] = 1.2*X[i ,0] + 0.75*np.sin(X[i,1]) + np.sin(X[i,2])  + noise


model = li.LinearRegression()
model.fit(X,Y)
o = model.predict(X)

mse = met.mean_squared_error(Y,o)
print(mse)
plt.scatter(Y[:,0] , o[:,0] , s = 10)
plt.xlabel('target')
plt.ylabel('predicted')
plt.plot([-3,+3],[-3,+3] , c = 'red',label = 'y=x')
plt.legend()
plt.show()