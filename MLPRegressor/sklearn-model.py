                ###################### SEBAXU #########################
                # This is my first project and I am proud of it ,     #
                # I feel hungry now I may free up later and edit it   # 
                # to become a ready-made library, but I'm tired becaus#
                # of the long week I spent with this matter           #
                # I feel really tired Let the ducks spread everywhere #
                ###################### SEBAXU #########################

from numpy import array
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# create the training data
x = array([[-10,  -9  ,-8 , -7 , -6,  -5  ,-4  ,-3 , -2 , -1  , 0  , 1 ,  2,   3  , 4 ,  5  , 6 ,  7,   8,  9]])
x = x.reshape(-1, 20) # Reshape x so it has 20 features
y = array([[14. , 15.8, 17.6 ,19.4, 21.2, 23. , 24.8 ,26.6, 28.4 ,30.2, 32.,  33.8, 35.6, 37.4, 39.2, 41. , 42.8 ,44.6 ,46.4 ,48.2]])
print(x)
# define and fit the model
mlp = MLPRegressor(hidden_layer_sizes=(100,100,100), activation='logistic', solver='lbfgs')
mlp.fit(x, y)

# use the model to make predictions
predictions = mlp.predict(x)


fig, ax = plt.subplots()
ax.plot(x.T, y.T, '-b', label='Real Temperatures')
ax.plot(x.T, predictions.T, '-r', label='Real Temperatures')

plt.show()

