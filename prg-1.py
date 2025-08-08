from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Input data: Area (sq.ft) and corresponding Price (in ₹ lakhs)
area = np.array([500, 1000, 1500, 2000, 2500]).reshape(-1, 1)
price = np.array([25, 50, 75, 100, 125])

# Train the Linear Regression model
model = LinearRegression().fit(area, price)

# Predict price for new input
x_new = 1800
y_pred = model.predict([[x_new]])
print(f"Predicted Price for {x_new} sq.ft: ₹{y_pred[0]:.2f} lakhs")

# Plot data and regression line
plt.scatter(area, price, color='blue', label='Data')
plt.plot(area, model.predict(area), color='red', label='Model')
plt.title("House Price Prediction")
plt.xlabel("Area (sq.ft)")
plt.ylabel("Price (lakhs ₹)")
plt.legend()
plt.grid()
plt.show()
