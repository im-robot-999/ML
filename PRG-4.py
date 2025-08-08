from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
def load_data():
    data = load_iris()
    return data.data, data.target, data.target_names

# Train and return model
def train_model(X, y):
    model = DecisionTreeClassifier()
    return model.fit(X, y)

# Predict & evaluate
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

# Main logic
X, y, target_names = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = train_model(X_train, y_train)
accuracy = evaluate_model(model, X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Predict one example
sample = [[5.1, 3.5, 1.4, 0.2]]
pred = model.predict(sample)[0]
print(f"Predicted class: {target_names[pred]}")

# Visualize tree
plt.figure(figsize=(10, 6))
plot_tree(model, filled=True, feature_names=load_iris().feature_names, class_names=target_names)
plt.title("Decision Tree - Iris Dataset")
plt.show()
