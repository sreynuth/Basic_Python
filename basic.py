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

data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["Paris", "London", "New York"]
}

df = pd.DataFrame(data)
print(df)