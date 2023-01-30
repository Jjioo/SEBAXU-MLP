                ###################### SEBAXU #########################
                # This is my first project and I am proud of it ,     #
                # I feel hungry now I may free up later and edit it   # 
                # to become a ready-made library, but I'm tired becaus#
                # of the long week I spent with this matter           #
                # I feel really tired Let the ducks spread everywhere #
                ###################### SEBAXU #########################

from numpy import array
from sklearn.neural_network import MLPClassifier

# create the training data
inputs = array([[2], [1],[4]])
outputs = array([0, 1,0])

# define and fit the model
mlp = MLPClassifier(hidden_layer_sizes=(2,2,5), activation='logistic', solver='lbfgs')
mlp.fit(inputs, outputs)

# use the model to make predictions
for _ in range(5):
  predictions = mlp.predict([[_]])
  print(_,predictions) 