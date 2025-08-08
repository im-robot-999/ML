from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
def load_data():
    data = load_breast_cancer()
    return data.data, data.target, data.target_names

# Train model
def train_model(X, y):
    return GaussianNB().fit(X, y)

# Evaluate model
def evaluate(model, X_test, y_test, labels):
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Report:\n", classification_report(y_test, preds, target_names=labels))

# Predict new sample
def predict_sample(model, sample, labels):
    pred = model.predict([sample])[0]
    print("Predicted Class:", labels[pred])

# Main logic
X, y, label_names = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = train_model(X_train, y_train)
evaluate(model, X_test, y_test, label_names)
predict_sample(model, X_test[0], label_names)
