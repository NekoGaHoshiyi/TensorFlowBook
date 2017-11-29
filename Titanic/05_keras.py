import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from data_processing import get_test_data
from data_processing import get_train_data

train_data = get_train_data()
X = train_data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Child',
                'EmbarkedF', 'DeckF', 'TitleF', 'Honor']].as_matrix()
Y = train_data[['Deceased', 'Survived']].as_matrix()

# arguments that can be set in command line
tf.app.flags.DEFINE_integer('epochs', 10, 'Training epochs')
FLAGS = tf.app.flags.FLAGS

model = Sequential()
model.add(Dense(100, input_dim=11, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
adam = Adam(lr=0.001)
# compile the model and save weights only when there is an improvement
model.compile(loss='binary_crossentropy', optimizer=adam,
              metrics=['accuracy'])

model.fit(X, Y, epochs=FLAGS.epochs, validation_split=0.2)
loss_and_metrics = model.evaluate(X, Y, batch_size=128)

test_data = get_test_data()
X = test_data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Child',
               'EmbarkedF', 'DeckF', 'TitleF', 'Honor']].as_matrix()
classes = model.predict(X, batch_size=128)
predictions = np.argmax(classes, 1)

submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": predictions
})

submission.to_csv("titanic-submission.csv", index=False)
