# write your code here
import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier

# Load the dataset

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
x_train_flattened = x_train.reshape(len(x_train), 28*28)

n_rows= 6000
features = np.concatenate((x_train, x_test))
target = np.concatenate((y_train, y_test))

x_train, x_test, y_train, y_test = train_test_split(features[:n_rows], target[:n_rows], test_size=0.3, random_state=40)
x_train_flatten = x_train.reshape(len(x_train), 28*28)
x_test_flatten = x_test.reshape(len(x_test), 28*28)


scores = {}
models = [

    KNeighborsClassifier(),

    DecisionTreeClassifier(random_state=40),

    LogisticRegression(random_state=40),

    RandomForestClassifier(random_state=40)

]

def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    model.fit(features_train, target_train)
    y_pred = model.predict(features_test)
    score = accuracy_score(y_test, y_pred)
    scores[model] = score
    print(f'Model: {model}\nAccuracy: {score}\n')


#print(f"The answer to the question: {best_model.__class__.__name__} - {best_score:.3f}")

x_train_norm = Normalizer().transform(x_train_flatten)
x_test_norm = Normalizer().transform(x_test_flatten)

#for model in models:
    #fit_predict_eval(model, x_train_norm, x_test_norm, y_train, y_test)


#sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
#best_model, best_score = sorted_scores[0]
#second_best_model, second_best_score = sorted_scores[1]
#print(f"The answer to the 1st question: yes\n")
#print(f"The answer to the 2nd question: {best_model.__class__.__name__}-{best_score:.3f}, {second_best_model.__class__.__name__}-{second_best_score:.3f}")

def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    # here you fit the model
    model.fit(features_train, target_train)
    # make a prediction
    y_pred = model.predict(features_test)
    # calculate accuracy and save it to score
    score = accuracy_score(y_test, y_pred)
    return score


knn = KNeighborsClassifier()
rf = RandomForestClassifier(random_state=40)

knn_parameters = dict(n_neighbors=[3, 4], weights=['uniform', 'distance'], algorithm=['auto', 'brute'])
rfc_parameters = dict(n_estimators=[300, 500], max_features=['auto', 'log2'], class_weight=['balanced', 'balanced_subsample'])

knn_clf = GridSearchCV(knn, knn_parameters, scoring='accuracy', n_jobs=-1)
knn_clf.fit(x_train_norm, y_train)

rfc_clf = GridSearchCV(rf, rfc_parameters, scoring='accuracy', n_jobs=-1)
rfc_clf.fit(x_train_norm, y_train)

print("K-nearest neighbours algorithm")
print(f"best estimator: {knn_clf.best_estimator_}")

print(f"accuracy: {fit_predict_eval(knn_clf, x_train_norm, x_test_norm, y_train, y_test)}\n")

print("Random forest algorithm")
print(f"best estimator: {rfc_clf.best_estimator_}")
print(f"accuracy: {np.sqrt(fit_predict_eval(rfc_clf, x_train_norm, x_test_norm, y_train, y_test))}\n")
