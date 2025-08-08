# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 12:07:43 2025

@author: Admin
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
def load_data():
    data = load_iris()
    return data.data, data.target

# Create base classifiers
def create_models():
    svm = SVC(probability=True)
    knn = KNeighborsClassifier(n_neighbors=3)
    tree = DecisionTreeClassifier()
    return svm, knn, tree

# Build voting classifier
def build_voting_model(models):
    return VotingClassifier(estimators=[
        ('svm', models[0]),
        ('knn', models[1]),
        ('tree', models[2])
    ], voting='soft')  # 'soft' uses predicted probabilities

# Train and evaluate model
def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))

# Main logic
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

models = create_models()
voting_model = build_voting_model(models)
evaluate(voting_model, X_train, X_test, y_train, y_test)
