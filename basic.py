# Variables
# name = "Alice"       # string
# age = 25             # integer
# height = 5.6         # float
# is_student = True    # boolean

# print(name, age, height, is_student)

# # For Loops
# for i in range(5):  # 0 to 4
#     print("Iteration:", i)


# # While Loops
# count = 0
# while count < 5:
#     print("Count:", count)
#     count += 1

# # Functions
# def greet(name):
#     return f"Hello, {name}!"

# print(greet("Alice"))
# print(greet("Bob"))

# # List
# fruits = ["apple", "banana", "cherry"]
# print(fruits[0])       # Access first item "apple"
# fruits.append("mango") # Add item "['apple', 'banana', 'cherry', 'mango']"
# print(fruits)

# for fruit in fruits:   # Loop through list
#     print(fruit)

# # Dictionaries
# person = {
#     "name": "Alice",
#     "age": 25,
#     "city": "Paris"
# }
# print(person["name"])        # Access by key
# person["age"] = 26           # Update value
# person["email"] = "a@x.com"  # Add new key-value

# for key, value in person.items():
#     print(key, ":", value)

import numpy as np
 
# # Creating arrays
# arr = np.array([1, 2, 3, 4, 5])
# print(arr)            # [1 2 3 4 5]
# print(type(arr))      # <class 'numpy.ndarray'>

# #Special arrays
# zeros = np.zeros((2, 3))      # 2x3 matrix of 0s
# ones = np.ones((3, 2))        # 3x2 matrix of 1s
# identity = np.eye(3)          # 3x3 Identity matrix
# range_arr = np.arange(0, 10, 2)  # [0 2 4 6 8]
# linspace_arr = np.linspace(0, 1, 5)  # [0. 0.25 0.5 0.75 1.]

# # Array Operations
# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# print(a + b)   # [5 7 9]
# print(a * b)   # [4 10 18]
# print(a ** 2)  # [1 4 9]
# print(np.sqrt(a))  # [1.         1.41421356 1.73205081]

# # Matrix Operations
# A = np.array([[1, 2],
#               [3, 4]])

# B = np.array([[5, 6],
#               [7, 8]])

# print(A + B)          # Matrix addition
# print(A @ B)          # Matrix multiplication (dot product)
# print(np.dot(A, B))   # Same as above
# print(A.T)            # Transpose

# # ==> Note: @ and dot
# [ [1*5 + 2*7,  1*6 + 2*8],
#   [3*5 + 4*7,  3*6 + 4*8] ]
# = [ [19, 22],
#     [43, 50] ]

# # Useful Functions
# arr = np.array([[1, 2, 3],
#                 [4, 5, 7]])

# print(arr.shape)   # (2, 3)
# print(arr.size)    # 6 (elements)
# print(arr.ndim)    # 2 (dimensions)
# print(arr.sum())   # 21
# print(arr.mean())  # 3.5 (sum() / len())
# print(arr.max())   # 7 # If word check # ["apple", "banana", "kiwi"] ==> banana (longest word)
# print(arr.min())   # 1

import pandas as pd

# data = {
#     "Name": ["Alice", "Bob", "Charlie"],
#     "Age": [25, 30, 35],
#     "City": ["Paris", "London", "New York"]
# }

# df = pd.DataFrame(data)
# print(df)
# print(df.head())      # first 5 rows
# print(df.tail())      # last 5 rows
# print(df.shape)       # (rows, columns)
# print(df.columns)     # column names
# print(df.info())      # summary
# print(df.describe())  # statistics for numeric columns
# print(df["Name"])            # single column
# print(df[["Name", "Age"]])   # multiple columns
# print(df.iloc[0])            # first row (by index)
# print(df.iloc[0:2])          # first 2 rows
# print(df.loc[0, "Name"])     # specific cell

# print(df[df["Age"] > 25])            # filter by condition
# print(df[df["City"] == "London"])    # filter by value

# df["Age_plus_10"] = df["Age"] + 10
# print(df)

# df.to_csv("output1.csv", index=False)
# df.to_excel("output.xlsx", index=False)

# # Load Titanic dataset (example)
# data = pd.read_csv("titanic.csv")
# print(data.head())

# # Unzip Titanic dataset
# import zipfile
# with zipfile.ZipFile("titanic.zip", "r") as zip_ref:
#     zip_ref.extractall("titanic")

# # Load CSV into pandas
# data = pd.read_csv("titanic.csv")
# print(data.head())

# # Plot graphs: histograms, bar charts, scatter plots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# data = pd.DataFrame({
#     "Age": [22, 25, 30, 35, 40, 45, 50, 55, 60, 65],
#     "Salary": [2000, 2500, 2700, 3200, 4000, 4200, 5000, 5200, 6000, 6500],
#     "Department": ["HR", "IT", "IT", "Finance", "Finance", "HR", "IT", "Finance", "HR", "IT"]
# })
# plt.hist(data["Age"], bins=5, color="skyblue", edgecolor="black")
# plt.title("Histogram of Age")
# plt.xlabel("Age")
# plt.ylabel("Count")
# plt.show()
# sns.countplot(x="Department", data=data, palette="pastel")
# plt.title("Employees per Department")
# plt.show()
# dept_salary = data.groupby("Department")["Salary"].mean()

# dept_salary.plot(kind="bar", color="orange", edgecolor="black")
# plt.title("Average Salary by Department")
# plt.xlabel("Department")
# plt.ylabel("Average Salary")
# plt.show()

# plt.scatter(data["Age"], data["Salary"], color="purple")
# plt.title("Scatter Plot: Age vs Salary")
# plt.xlabel("Age")
# plt.ylabel("Salary")
# plt.show()

# sns.scatterplot(x="Age", y="Salary", hue="Department", data=data, palette="Set2")
# plt.title("Scatter Plot with Departments")
# plt.show()


# Example dataset
# data = pd.DataFrame({
#     "Size": [1400, 1600, 1700, 1875, 1100],
#     "Bedrooms": [3, 3, 4, 4, 2],
#     "Location": ["Phnom Penh", "Siem Reap", "Phnom Penh", "Battambang", "Siem Reap"],
#     "Price": [340000, 360000, 400000, 420000, 250000]
# })

# # Features (X) = inputs
# X = data[["Size", "Bedrooms", "Location"]]

# # Label (y) = output
# y = data["Price"]

# print("Features (X):")
# print(X)

# print("\nLabel (y):")
# print(y)

# # Sample dataset with missing values
data = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "David", None],
    "Age": [25, None, 30, 35, 40],
    "Salary": [50000, 60000, None, 80000, 90000]
})

# print(data)
# print(data.isnull())        # Boolean mask of NaN
# print(data.isnull().sum())  # Count missing per column

# # Drop rows with any NaN
# data_drop_rows = data.dropna()

# # Drop columns with any NaN
# data_drop_cols = data.dropna(axis=1)

# print(data_drop_rows)

# # (a) Fill with a constant
# data_fill_const = data.fillna(0)
# print(data_fill_const)

# # (b) Fill with mean / median / mode
# data["Age"] = data["Age"].fillna(data["Age"].mean())     # Fill with mean (mean = Sum of all values​/Number of values)
# data["Salary"] = data["Salary"].fillna(data["Salary"].median())  # Fill with median (sorted in order)
# print(data["Age"])
# print(data["Salary"])


# # (c) Forward fill / backward fill
# data_ffill = data.fillna(method="ffill")  # Forward fill
# print(data_ffill)
# data_bfill = data.fillna(method="bfill")  # Backward fill
# print(data_bfill)

# # Drop rows where specific column is NaN
# data_clean = data.dropna(subset=["Name"])  # Remove rows where Name is missing
# print(data_clean)

# # Line Plot
# x = [1, 2, 3, 4, 5]
# y = [2, 4, 6, 8, 10]

# plt.plot(x, y, color="blue", marker="o")
# plt.title("Line Plot Example")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.show()

# # Bar Chart
# categories = ["A", "B", "C", "D"]
# values = [10, 20, 15, 25]

# plt.bar(categories, values, color="orange")
# plt.title("Bar Chart Example")
# plt.xlabel("Category")
# plt.ylabel("Value")
# plt.show()

# # Histogram
# data = [7, 8, 5, 6, 8, 10, 7, 9, 5, 6, 7, 8, 10, 9, 6]

# plt.hist(data, bins=5, color="skyblue", edgecolor="black")
# plt.title("Histogram Example")
# plt.xlabel("Bins")
# plt.ylabel("Frequency")
# plt.show()

# # Scatter Plot
# x = [5, 7, 8, 7, 6, 9, 5, 6]
# y = [99, 86, 87, 88, 100, 86, 103, 87]

# plt.scatter(x, y, color="purple")
# plt.title("Scatter Plot Example")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()

# # Pie Chart
# sizes = [40, 30, 20, 10]
# labels = ["A", "B", "C", "D"]

# plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=["gold", "lightblue", "lightgreen", "pink"])
# plt.title("Pie Chart Example")
# plt.show()

# # Multiple Plots in One Figure
# x = [1, 2, 3, 4, 5]
# y1 = [2, 4, 6, 8, 10]
# y2 = [1, 2, 3, 4, 5]

# plt.plot(x, y1, label="y = 2x")
# plt.plot(x, y2, label="y = x", linestyle="--")

# plt.title("Multiple Lines")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend()
# plt.show()

