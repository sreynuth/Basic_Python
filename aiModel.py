# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# # ==> supervised
# # Training data (X = size of house in square feet, y = price in $1000s)
# X = [[1000], [1500], [2000], [2500]]   # features
# y = [200, 250, 300, 350]               # labels

# # Create and train the model
# model = LinearRegression()
# model.fit(X, y)

# Predict the price of a new house
# prediction = model.predict([[1800]])
# print("Predicted price:", prediction[0])  # Predicted price: 280.0 (0.1 x 1800 + 100)

# # ==> Train/Test split
# Data
# X = [[1000], [1500], [2000], [2500], [3000], [3500]]
# y = [200, 250, 300, 350, 400, 450]

# # Split data: 80% train, 20% test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Predict on test set
# y_pred = model.predict(X_test)

# # Compare results
# print("X_test:", X_test)
# print("y_test (real):", y_test)
# print("y_pred (predicted):", y_pred)

# # Evaluate error
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", mse)

# # ==> Load Titanic dataset (example from seaborn)
# import seaborn as sns
# data = sns.load_dataset('titanic')

# # Keep only some columns and drop missing values for simplicity
# data = data[['survived', 'pclass', 'sex', 'age', 'fare']].dropna()

# # Convert categorical column 'sex' to numbers
# data['sex'] = data['sex'].map({'male': 0, 'female': 1})

# # Features and labels
# X = data[['pclass', 'sex', 'age', 'fare']]
# y = data['survived']

# # Train/Test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
# # Train a classifier
# # Logistic Regression (good for binary classification)
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Make predictions
# y_pred = model.predict(X_test)

# # Accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# # Confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:\n", cm)

# # ==> Overfitting vs underfitting
# Sample data
# X = np.array([1,2,3,4,5,6,7,8,9]).reshape(-1,1)
# y = np.array([1.2,1.9,3.2,3.9,5.1,5.8,7.1,7.9,9.2])

# # Underfitting (degree=1)
# poly1 = PolynomialFeatures(degree=1)
# X_poly1 = poly1.fit_transform(X)
# model1 = LinearRegression().fit(X_poly1, y)
# y_pred1 = model1.predict(X_poly1)

# # Overfitting (degree=8)
# poly8 = PolynomialFeatures(degree=8)
# X_poly8 = poly8.fit_transform(X)
# model8 = LinearRegression().fit(X_poly8, y)
# y_pred8 = model8.predict(X_poly8)

# # Plot
# plt.scatter(X, y, color='black', label='Data')
# plt.plot(X, y_pred1, label='Underfit', color='blue')
# plt.plot(X, y_pred8, label='Overfit', color='red')
# plt.legend()
# plt.show()

# # ==> Accuracy, precision, recall
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load dataset
import seaborn as sns
data = sns.load_dataset('titanic')
data = data[['survived', 'pclass', 'sex', 'age', 'fare']].dropna()
data['sex'] = data['sex'].map({'male': 0, 'female': 1})

X = data[['pclass', 'sex', 'age', 'fare']]
y = data['survived']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)