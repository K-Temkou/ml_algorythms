import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model, preprocessing

data = pd.read_csv(r"C:\Users\Have Fun\Documents\coding\python\machine_learning\data\car.data")
#print(data.head())

# converting string values into numerical values our algorythm can work with
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

#print(safety)

predict = "class"
X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

# test_train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=1234)

#print(X_train, y_test)

model = KNeighborsClassifier(n_neighbors=7)

model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(acc)

names = ["unacc", "acc", "good", "vgood"]


predicted = model.predict(X_test)
for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", X_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([X_test[x]], 9)
    print("N: ", n)
