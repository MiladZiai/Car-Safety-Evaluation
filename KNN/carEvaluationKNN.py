import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing

data = pd.read_csv("car.data")
le = preprocessing.LabelEncoder()

#create a list with integer values for all features
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

#create tuples of features for each row
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

#split training and test set and set test size to 10%
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

#select the KNN model and set number of neighbors
model = KNeighborsClassifier(n_neighbors=9)

#train the model and select accuracy
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)

#predict the data
predicted = model.predict(x_test)

#classifying from 0-3 where 0 = unacc etc
names = ["unacc", "acc", "good", "veryGood"]

#print predicted and actual data
for x in range(len(x_test)):
    print("predicted: ", names[predicted[x]], "data: ", x_test[x], " actual: ", names[y_test[x]])