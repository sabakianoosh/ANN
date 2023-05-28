import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPRegressor



def z(x,y):
    return np.exp(-(2*x*y + np.square(y)))


x = np.arange(-1,1,0.05)
xy = [(j,k) for j in x for k in x]
out = [z(p[0],p[1]) for p in xy]

x_train, x_test, y_train, y_test = train_test_split(xy, out)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


x1_vals = np.array([p[0] for p in x_train])
x2_vals = np.array([p[1] for p in x_train])


x1_valz = np.array([p[0] for p in x_test])
x2_valz = np.array([p[1] for p in x_test])


ax.scatter(x1_vals, x2_vals, y_train)
plt.show()

mlp = MLPRegressor(
    hidden_layer_sizes=[10],
    max_iter=100, 
    tol=0,
)

# train network
mlp.fit(x_train,y_train)

# test
predictions = mlp.predict(x_test)
mse = mean_squared_error(y_test, predictions)
print(mse)
ax.scatter(x1_valz, x2_valz, predictions, c='red')

plt.show()