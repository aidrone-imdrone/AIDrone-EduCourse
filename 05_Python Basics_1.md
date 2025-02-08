# Python Basic Education

Python is a powerful programming language that is easy to learn. This lecture material covers basic concepts and practical exercises to help students understand Python easily.

## 📌 1. Introduction to Python
### ✅ What is Python?
- A programming language with simple and concise syntax
- Used in various fields such as AI, data analysis, web development, and game development

### ✅ Features of Python
- Simple and readable syntax
- Free and provides various libraries
- Runs on multiple operating systems (Windows, Mac, Linux)

### ✅ Installing and Running Python
- **Google Colab** (Runs online instantly)
- **Jupyter Notebook** (Widely used for data analysis)
- **Run Python in the terminal after installation** (Type `python`)

---

## 📌 2. Basic Python Syntax

### 🔹 Variables and Data Types
```python
name = "Alice"  # String
age = 15        # Integer
height = 165.5  # Float
is_student = True  # Boolean

print(name, age, height, is_student)
```

### 🔹 Input and Output
```python
name = input("Enter your name: ")
print("Hello,", name, "!")
```

### 🔹 Operators
```python
a = 10
b = 3
print(a + b)  # Addition
print(a - b)  # Subtraction
print(a * b)  # Multiplication
print(a / b)  # Division (float)
print(a // b) # Division (integer)
print(a % b)  # Modulus
print(a ** b) # Exponentiation (10^3)
```

### 🔹 Conditional Statements (if statements)
```python
age = int(input("Enter your age: "))

if age >= 18:
    print("You are an adult.")
elif age >= 13:
    print("You are a teenager.")
else:
    print("You are a child.")
```

### 🔹 Loops (for, while)
```python
for i in range(1, 6):
    print(i)

num = 1
while num <= 5:
    print(num)
    num += 1
```

### 🔹 Lists
```python
fruits = ["Apple", "Banana", "Strawberry"]
print(fruits[0])  # Prints "Apple"
fruits.append("Grapes")  # Adds an item to the list
print(fruits)

for fruit in fruits:
    print(fruit)
```

---

## 📌 3. Functions
```python
def greet(name):
    print("Hello,", name, "!")

greet("Chulsoo")
```

---

## 📌 4. Practice Exercises

### ✅ Problem 1: Compare Two Numbers
```python
num1 = int(input("First number: "))
num2 = int(input("Second number: "))

if num1 > num2:
    print("The first number is larger.")
elif num1 < num2:
    print("The second number is larger.")
else:
    print("Both numbers are equal.")
```

### ✅ Problem 2: Multiplication Table
```python
dan = int(input("Enter the multiplication table number: "))

for i in range(1, 10):
    print(dan, "x", i, "=", dan * i)
```

### ✅ Problem 3: Calculate the Average of a List
```python
numbers = [10, 20, 30, 40, 50]
sum_numbers = sum(numbers)
average = sum_numbers / len(numbers)

print("Average:", average)
```

---

## 📌 5. Python Application Examples

### ✅ 1. Simple Calculator
```python
def calculator(a, b, op):
    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        return a / b
    else:
        return "Invalid operator."

num1 = int(input("First number: "))
num2 = int(input("Second number: "))
operator = input("Operator (+,-,*,/): ")

result = calculator(num1, num2, operator)
print("Result:", result)
```

### ✅ 2. Number Guessing Game
```python
import random

secret = random.randint(1, 100)
guess = 0

while guess != secret:
    guess = int(input("Guess a number between 1 and 100: "))

    if guess < secret:
        print("Enter a larger number.")
    elif guess > secret:
        print("Enter a smaller number.")
    else:
        print("Correct!")
```

---

## 📌 6. Summary 
### ✅ Summary of Today's Lesson
- Variables, data types, input/output
- Conditional statements, loops
- Lists, functions
- Writing simple Python programs


