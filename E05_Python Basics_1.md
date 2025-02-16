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

## 📌 2. Python Installation and Usage Guide

## 1. Installing Python

### Installing on Windows
1. **Download Python**
   - Visit the official Python website: [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - Click the "Download Python" button to download the latest version.
   
2. **Run the Installer**
   - Open the downloaded `.exe` file.
   - **Check the option**: `Add Python to PATH`
   - Click `Install Now`

<br/>

<img src="https://github.com/user-attachments/assets/d3cec976-b54c-4e8f-b4b1-000ba07fa78a" width="500">

<br/>

3. **Verify Installation**
   - Open Command Prompt (`Win + R`, type `cmd`, then press Enter)
   - Enter the following command:
     ```sh
     python --version
     ```
   - If the installed Python version is displayed, the installation was successful.

<br/>
  
<img src="https://github.com/user-attachments/assets/1acf2bdb-37a2-4c4b-896b-9a2711120676" width="500">

<br/><br/>

### Installing on macOS
1. **Check if Python is already installed**
   - Open Terminal and run:
     ```sh
     python3 --version
     ```
   - If Python is not installed, proceed to the next step.

2. **Install Python using Homebrew** (Recommended)
   - If Homebrew is not installed, run:
     ```sh
     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
     ```
   - Install Python:
     ```sh
     brew install python
     ```

3. **Verify Installation**
   - Run:
     ```sh
     python3 --version
     ```

### Installing on Linux
1. **Check if Python is already installed**
   - Run:
     ```sh
     python3 --version
     ```
   
2. **Install Python on Debian/Ubuntu**
   - Run the following commands:
     ```sh
     sudo apt update
     sudo apt install python3 python3-pip
     ```
   
3. **Install Python on CentOS/RHEL**
   - Run:
     ```sh
     sudo yum install python3
     ```
   
4. **Verify Installation**
   - Run:
     ```sh
     python3 --version
     ```

## 2. Using Python

### Running Python in the Terminal
- Open a terminal or command prompt.
- Enter:
  ```sh
  python
  ```
- The Python interpreter (`>>>`) will appear.
- To exit, enter:
  ```sh
  exit()
  ```

### IDE에서  Python 실행

<br/>

<img src="https://github.com/user-attachments/assets/6bddfdc5-c622-4479-b4af-16119ef2dd46" width="300">

<br/><br/>

<img src="https://github.com/user-attachments/assets/6b640997-175a-49a6-a3e5-37d33a535ed2" width="800">

<br/>

### Running a Python Script
1. Create a file named `hello.py` and write the following code:
   ```python
   print("Hello, World!")
   ```
2. Save the file and run it with:
   ```sh
   python hello.py  # or python3 hello.py
   ```

## 3. Installing Python Packages
- Install packages using `pip`:
  ```sh
  pip install package_name
  ```
- Example:
  ```sh
  pip install numpy
  ```
- Check installed packages:
  ```sh
  pip list
  ```
- Upgrade a package:
  ```sh
  pip install --upgrade package_name
  ```

## 4. Using Virtual Environments
- Create a virtual environment:
  ```sh
  python -m venv myenv
  ```
- Activate the virtual environment:
  - Windows:
    ```sh
    myenv\Scripts\activate
    ```
  - macOS/Linux:
    ```sh
    source myenv/bin/activate
    ```
- Deactivate the virtual environment:
  ```sh
  deactivate
  ```

## 5. Writing and Running a Simple Python Program
1. Create a file named `example.py` and write the following code:
   ```python
   name = input("Enter your name: ")
   print(f"Hello, {name}!")
   ```
2. Run the script with:
   ```sh
   python example.py
   ```
<br/>

## 📌3. Basic Python Syntax

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


