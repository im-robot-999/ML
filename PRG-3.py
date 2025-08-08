from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

# Sample Data: Marks vs Pass(1)/Fail(0)
marks = np.array([[35], [45], [50], [60], [70], [85]])
result = np.array([0, 0, 1, 1, 1, 1])  # 0 = Fail, 1 = Pass

# Train model
model = LogisticRegression().fit(marks, result)

# Prediction function
def predict_pass(mark):
    prob = model.predict_proba([[mark]])[0][1]
    status = "Pass" if prob >= 0.5 else "Fail"
    return status, prob

# Test prediction
m = 48
status, probability = predict_pass(m)
print(f"Mark: {m} → {status} (Prob: {probability:.2f})")

# Visualize result
x_vals = np.linspace(30, 90, 100).reshape(-1, 1)
y_probs = model.predict_proba(x_vals)[:, 1]

plt.scatter(marks, result, c='blue', label='Actual')
plt.plot(x_vals, y_probs, 'r', label='Logistic Curve')
plt.axvline(m, color='green', linestyle='--', label=f'Test Mark = {m}')
plt.title("Pass/Fail Prediction using Logistic Regression")
plt.xlabel("Marks")
plt.ylabel("Probability of Passing")
plt.legend()
plt.grid()
plt.show()
