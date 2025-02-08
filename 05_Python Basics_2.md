# Python File Input/Output (File I/O)

File Input/Output (File I/O) allows programs to read and write files. In Python, you can manipulate files using the **open() function**.

---

## 1. Opening and Closing Files

In Python, the `open()` function is used to open files.

### 🔹 File Opening Modes
| Mode | Description |
|------|------------|
| `r`  | Read mode (file must exist) |
| `w`  | Write mode (creates file if not exists, overwrites existing content) |
| `a`  | Append mode (creates file if not exists, retains existing content) |
| `x`  | Creates a new file (raises an error if file already exists) |
| `rb` | Binary read mode |
| `wb` | Binary write mode |

```python
# Opening and closing a file
file = open("example.txt", "w")  # Open file in write mode
file.write("Hello, Python!")  # Write data to file
file.close()  # Close the file
```

✅ You must call `close()` to ensure data is saved properly.

---

## 2. Writing to a File (write)
```python
with open("example.txt", "w") as file:
    file.write("This is the first line.
")
    file.write("This is the second line.
")
```

✅ Using `with open()` automatically calls `close()`

---

## 3. Reading from a File (read)
```python
with open("example.txt", "r") as file:
    content = file.read()  # Read entire file
    print(content)
```

✅ `read()` reads the entire file content as a string.

```python
with open("example.txt", "r") as file:
    lines = file.readlines()  # Read all lines into a list
    print(lines)
```

✅ `readlines()` returns a list of all lines.

---

## 4. Reading Line by Line (readline)
```python
with open("example.txt", "r") as file:
    line = file.readline()  # Read one line at a time
    while line:
        print(line.strip())  # Remove newline character and print
        line = file.readline()
```

✅ `readline()` reads one line at a time.

---

## 5. Appending to a File (append)
```python
with open("example.txt", "a") as file:
    file.write("This is an additional line!
")
```

✅ Using mode `a` preserves existing content and appends new data.

---

## 6. Checking if a File Exists
```python
import os

if os.path.exists("example.txt"):
    print("The file exists.")
else:
    print("The file does not exist.")
```

✅ Use `os.path.exists()` to check if a file exists.

---

## 7. Reading & Writing CSV Files
### ✅ Writing a CSV File
```python
import csv

data = [
    ["Name", "Age", "Job"],
    ["Chulsoo", 20, "Student"],
    ["Younghee", 25, "Developer"]
]

with open("data.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(data)  # Write multiple rows
```

### ✅ Reading a CSV File
```python
with open("data.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)
```

✅ Use `csv.reader()` to read CSV data as a list.

---

## 8. Reading & Writing JSON Files
### ✅ Writing a JSON File
```python
import json

data = {"Name": "Chulsoo", "Age": 20, "Job": "Student"}

with open("data.json", "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
```

### ✅ Reading a JSON File
```python
with open("data.json", "r", encoding="utf-8") as file:
    data = json.load(file)
    print(data)
```

✅ `json.dump()` saves JSON data to a file, and `json.load()` reads JSON files into Python objects.

---

## 9. Example: Simple Notepad Program
```python
def write_note():
    note = input("Enter your note: ")
    with open("notes.txt", "a") as file:
        file.write(note + "
")
    print("Note saved successfully!")

def read_notes():
    with open("notes.txt", "r") as file:
        print("=== Saved Notes ===")
        print(file.read())

while True:
    choice = input("Write Note (1) / Read Notes (2) / Exit (0): ")
    if choice == "1":
        write_note()
    elif choice == "2":
        read_notes()
    elif choice == "0":
        break
    else:
        print("Invalid input.")
```

✅ A simple notepad program that allows users to save and retrieve notes.

---

# 🚀 Summary
- **Opening files (`open()`)**
- **Writing to files (`write()`)**
- **Reading files (`read()`, `readline()`, `readlines()`)**
- **Appending to files (`append`)**
- **Checking file existence (`os.path.exists()`)**
- **Handling CSV & JSON files**
- **A simple notepad program using file handling**

Mastering Python File I/O allows you to handle various data processing tasks efficiently! 🚀

