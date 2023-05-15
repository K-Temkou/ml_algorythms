import torch
import torch.nn as nn
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
import pickle

# 1) forward pass, input_size, output_size
# 2) (get gradient) loss and optimizer
# 3) training loop
#       forward pass
#       get gradient
#       update weights
#       empty gradients


data = pd.read_csv(r"C:\Users\Have Fun\Documents\coding\python\machine_learning\data\student\student-mat.csv", sep=";")

predict = data[["G3"]]

# defining the parameters we want to use to predict the final grade
data = data[["G1", "G2", "studytime", "absences", "failures"]]

print(data.head())
print("G3")



X = np.array(data)
y = np.array(predict)

# test_train split
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# creating the model
linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)

acc = linear.score(x_test, y_test)

print(acc)

with open("student_matsave.pickle", "wb") as f:
            pickle.dump(linear, f)


prediction = linear.predict(x_test)


# pltting the data into a graph
p = "G2"
style.use("ggplot")
plt.scatter(data[p], predict["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()
