# **Raspberry Pi Linux Education Guide**

## **1. Introduction**
### **Objective**
This guide is designed to teach students the fundamentals of Linux using a Raspberry Pi. The course covers essential Linux commands, file management, networking, and scripting, providing hands-on experience with Raspberry Pi's operating system.

### **Prerequisites**
- Basic knowledge of computers
- A Raspberry Pi with Raspberry Pi OS installed
- A keyboard, mouse, and monitor (or SSH access)

---

## **2. Getting Started with Linux on Raspberry Pi**
### **Logging into the Raspberry Pi**
#### **Using a Monitor:**
- Default user: `pi`
- Default password: `raspberry`

#### **Using SSH:**
```bash
ssh pi@raspberrypi.local
```
(Default password: `raspberry`)

### **Updating the System**
Before starting, update the Raspberry Pi OS:
```bash
sudo apt update && sudo apt upgrade -y
```

---

## **3. Basic Linux Commands**
### **Navigating the Filesystem**
```bash
pwd        # Show current directory
ls         # List files and directories
cd /home/pi  # Change directory
```

### **File Management**
```bash
touch file.txt        # Create a new file
mkdir my_folder       # Create a new directory
mv file.txt my_folder/  # Move file to a folder
rm file.txt           # Delete a file
rmdir my_folder       # Remove an empty folder
rm -r my_folder       # Remove a folder and its contents
```

### **Viewing and Editing Files**
```bash
cat file.txt   # View file contents
nano file.txt  # Edit file with nano editor
```

---

## **4. User and Permission Management**
### **Creating and Managing Users**
```bash
sudo adduser student  # Create a new user
sudo passwd student   # Change password
sudo userdel student  # Delete a user
```

### **File Permissions**
```bash
ls -l file.txt  # Show file permissions
chmod 755 file.txt  # Change permissions
chown pi:pi file.txt  # Change file ownership
```

---

## **5. Networking and Remote Access**
### **Checking Network Information**
```bash
ifconfig  # Show network details (older systems)
ip a       # Show network details (newer systems)
ping google.com  # Test internet connection
```

### **Setting Up SSH for Remote Access**
Enable SSH:
```bash
sudo raspi-config
# Navigate to 'Interfacing Options' -> 'SSH' -> Enable
```
Connect via SSH:
```bash
ssh pi@raspberrypi.local
```

---

## **6. Shell Scripting Basics**
### **Creating a Simple Script**
```bash
echo "Hello, World!" > hello.sh
chmod +x hello.sh
./hello.sh
```

### **Example: Automated Backup Script**
```bash
#!/bin/bash
mkdir -p ~/backup
cp -r ~/Documents/* ~/backup/
echo "Backup completed!"
```
Save the script as `backup.sh`, then run:
```bash
chmod +x backup.sh
./backup.sh
```

---

## **7. Advanced Topics**
### **Installing and Managing Software**
```bash
sudo apt install htop  # Install software
htop  # Run htop (system monitor)
sudo apt remove htop  # Remove software
```

### **Process Management**
```bash
ps aux  # List running processes
kill PID  # Kill a process (replace PID with actual process ID)
```

---

## **8. Conclusion and Further Learning**
### **Next Steps**
- Learn more about shell scripting
- Explore Python programming on Raspberry Pi
- Set up a web server with Apache or Nginx

This guide provides a solid foundation in Linux basics on Raspberry Pi. Keep practicing and exploring new commands!

