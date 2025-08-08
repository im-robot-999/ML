from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
def load_data():
    iris = load_iris()
    return iris.data, iris.target, iris.target_names

# Train model
def train_model(X, y, kernel_type='linear'):
    return SVC(kernel=kernel_type, probability=True).fit(X, y)

# Evaluate model
def evaluate(model, X_test, y_test, labels):
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Report:\n", classification_report(y_test, preds, target_names=labels))

# Predict new sample
def predict_sample(model, sample, labels):
    pred = model.predict([sample])[0]
    prob = model.predict_proba([sample])[0]
    print(f"Predicted Class: {labels[pred]}")
    print("Class Probabilities:", {labels[i]: f"{p:.2f}" for i, p in enumerate(prob)})

# Main logic
X, y, label_names = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = train_model(X_train, y_train, kernel_type='linear')
evaluate(model, X_test, y_test, label_names)
predict_sample(model, [5.5, 2.5, 4.0, 1.3], label_names)
