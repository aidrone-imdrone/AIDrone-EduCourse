# Python Object-Oriented Programming (OOP)

Object-Oriented Programming (OOP) is a programming paradigm that structures a program using **classes and objects**. Python is a powerful object-oriented language that makes it easy to apply OOP concepts.

---

## 1. Object-Oriented Programming (OOP) Concepts
### ✅ Key Concepts of OOP
1. **Class**: A blueprint for creating objects.
2. **Object**: An instance created based on a class.
3. **Attribute (Variable)**: Data that an object holds (e.g., name, age).
4. **Method (Function)**: Actions that an object can perform.
5. **Constructor (`__init__`)**: A special method for initializing objects.
6. **Inheritance**: Extending the functionality of an existing class.
7. **Polymorphism**: The ability for the same method to behave differently in different classes.
8. **Encapsulation**: Restricting access to certain data to protect it.

---

## 2. Classes and Objects
### 🔹 Defining a Class and Creating an Object
```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age  

    def introduce(self):
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")

p1 = Person("Chulsoo", 20)
p2 = Person("Younghee", 25)

p1.introduce()
p2.introduce()
```

---

## 3. Class Attributes and Instance Attributes
```python
class Car:
    wheels = 4

    def __init__(self, brand, color):
        self.brand = brand
        self.color = color

car1 = Car("Hyundai", "Red")
car2 = Car("BMW", "Black")

print(car1.wheels)  # 4
print(car2.wheels)  # 4
```

---

## 4. Class Methods and Static Methods
```python
class Student:
    school = "Python High School"

    @classmethod
    def get_school(cls):
        return cls.school

print(Student.get_school())
```

```python
class MathUtils:
    @staticmethod
    def add(a, b):
        return a + b

print(MathUtils.add(10, 20))
```

---

## 5. Inheritance
```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        print(f"{self.name} makes a sound.")

class Dog(Animal):
    def speak(self):
        print(f"{self.name} barks.")

dog = Dog("Buddy")
dog.speak()
```

---

## 6. Polymorphism
```python
class Bird:
    def speak(self):
        print("Chirp!")

class Cat:
    def speak(self):
        print("Meow!")

def make_sound(animal):
    animal.speak()

b = Bird()
c = Cat()

make_sound(b)
make_sound(c)
```

---

## 7. Encapsulation
```python
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance

    def deposit(self, amount):
        self.__balance += amount

    def get_balance(self):
        return self.__balance

account = BankAccount(1000)
account.deposit(500)
print(account.get_balance())
```

---

## 📌 Comprehensive OOP Example: Student Management System
```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")

class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)
        self.student_id = student_id

    def introduce(self):
        print(f"I am a student, and my student ID is {self.student_id}.")

p = Person("Kim Chulsoo", 40)
s = Student("Lee Younghee", 20, "S12345")

p.introduce()
s.introduce()
```

---

# 🚀 Summary
- **Classes and Objects**
- **Inheritance, Polymorphism, Encapsulation**
- **Class Methods and Static Methods**
- **OOP Practical Examples**

Python's OOP enhances **code reusability** and **maintainability**, making it an essential concept to learn! 🚀

