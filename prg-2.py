from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Sample dataset (Area in sq.ft, Bedrooms) → Price in lakhs ₹
X = np.array([
    [1000, 2],
    [1200, 2],
    [1500, 3],
    [1800, 3],
    [2000, 4]
])
y = np.array([50, 55, 65, 75, 85])

# Train model
model = LinearRegression().fit(X, y)

# Predict for new house
def predict_price(area, rooms):
    return model.predict([[area, rooms]])[0]

# Example prediction
a, r = 1600, 3
print(f"Predicted price for {a} sq.ft, {r} BHK: ₹{predict_price(a, r):.2f} lakhs")

# 3D Plot (area vs price)
area_vals = X[:, 0]
plt.scatter(area_vals, y, c='blue', label='Data')
plt.plot(area_vals, model.predict(X), c='green', label='Prediction')
plt.title("Multiple Linear Regression")
plt.xlabel("Area (sq.ft)")
plt.ylabel("Price (lakhs ₹)")
plt.legend()
plt.grid()
plt.show()
