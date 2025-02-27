# Deep Learning Basics

## Table of Contents

1. **Overview of Deep Learning**
   - Relationship between AI, Machine Learning, and Deep Learning
   - History and Development of Deep Learning
   - Major Applications of Deep Learning

2. **Fundamentals of Neural Networks**
   - Neurons and Perceptron Model
   - Activation Functions
   - Feedforward Neural Networks

3. **Learning Principles of Deep Learning**
   - Loss Functions
   - Gradient Descent
   - Backpropagation Algorithm

4. **Types of Deep Learning Models**
   - CNN (Convolutional Neural Networks)
   - RNN (Recurrent Neural Networks)
   - Transformer Architecture

5. **Hands-on: Simple Image Classification**
   - Data Preparation and Preprocessing
   - Model Implementation and Training
   - Evaluation and Prediction

---

## 1. Overview of Deep Learning

<br/>

<img src="https://github.com/user-attachments/assets/fcadf4b2-aa60-4f1a-9dd7-0c7e34f8e3a0" width="700">

<br/><br/>

### Relationship between AI, Machine Learning, and Deep Learning

- **Artificial Intelligence (AI)**: The broad technology of mimicking human intelligence
- **Machine Learning (ML)**: A subfield of AI that learns patterns from data
- **Deep Learning (DL)**: A subset of machine learning using deep neural networks

### History and Development of Deep Learning

- 1943: McCulloch & Pitts propose artificial neuron models
- 1957: Rosenblatt introduces the Perceptron algorithm
- 1986: Development of the Backpropagation Algorithm
- 2006: Hinton coins the term "Deep Learning" and introduces Deep Belief Networks
- 2012: AlexNet wins the ImageNet competition, sparking the deep learning revolution
- 2014-Present: Various architectural advancements (GANs, Transformers, etc.)

### Major Applications of Deep Learning

- Computer Vision (Image Recognition, Object Detection)
- Natural Language Processing (Machine Translation, Chatbots, Text Summarization)
- Speech Recognition and Synthesis
- Recommendation Systems
- Game AI (AlphaGo, AlphaStar)
- Autonomous Vehicles
- Medical Diagnosis and Drug Development

---

## 2. Fundamentals of Neural Networks

<br/>

<img src="https://github.com/user-attachments/assets/a2988f40-4629-4925-8437-ba47ff252c00" width="800">

<br/><br/>

### Neurons and Perceptron Model

**Components of an Artificial Neuron**:
- Inputs (x₁, x₂, ..., xₙ)
- Weights (w₁, w₂, ..., wₙ)
- Bias (b)
- Weighted sum: z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
- Activation function: y = f(z)

**Perceptron Learning Method**:
- Make predictions based on inputs
- Compute error (Actual - Predicted)
- Update weights: w_new = w_old + α(Actual - Predicted) × Input

<br/>

<img src="https://github.com/user-attachments/assets/15e75733-2b62-413a-8557-75d262560549" width="800">

<br/><br/>

### Activation Functions

<br/>

<img src="https://github.com/user-attachments/assets/77957e1f-e39d-433e-ac97-9066acc6ad8b" width="800">

<br/><br/>

**Common Activation Functions**:

1. **Sigmoid**: f(x) = 1 / (1 + e^(-x))
   - Range: 0 to 1
   - Pros: Interpretable as probability
   - Cons: Vanishing gradient issue, slow computation

2. **Tanh**: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
   - Range: -1 to 1
   - Pros: Zero-centered, larger gradient than Sigmoid
   - Cons: Still suffers from vanishing gradients in deep networks

3. **ReLU**: f(x) = max(0, x)
   - Range: 0 to ∞
   - Pros: Efficient computation, reduces vanishing gradient issue
   - Cons: Dying ReLU problem (zero gradient for negative inputs)

4. **Leaky ReLU**: f(x) = max(α·x, x), where α is a small constant
   - Pros: Solves the Dying ReLU problem

### Feedforward Neural Networks

<br/>

<img src="https://github.com/user-attachments/assets/faa54ce3-18dc-44d7-9867-8b024aed0656" width="800">

<br/><br/>

- **Input Layer**: Feeds data into the network
- **Hidden Layers**: Learn complex features (deeper layers enhance abstraction)
- **Output Layer**: Produces final predictions

---

## 3. Learning Principles of Deep Learning

### Loss Functions

- Measure the difference between predicted and actual values
- The goal is to minimize the loss function

**Common Loss Functions**:

1. **Mean Squared Error (MSE)**: Used for regression tasks
   - L = (1/n) Σ(y - ŷ)²

2. **Cross-Entropy Loss**: Used for classification tasks
   - Binary Classification: L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
   - Multi-class Classification: L = -Σ y_i·log(ŷ_i)

### Gradient Descent

<br/>

<img src="https://github.com/user-attachments/assets/872b8f9e-3c01-447d-afd9-a780cd6bafa8" width="800">

<br/><br/>

**Basic Principle**:
- Update parameters in the direction that minimizes the loss function
- Compute the gradient of the loss function
- Parameter update formula: θ_new = θ_old - η·∇L(θ)

**Types of Gradient Descent**:
1. **Batch Gradient Descent**: Uses the entire dataset
2. **Stochastic Gradient Descent (SGD)**: Updates using one sample at a time
3. **Mini-batch Gradient Descent**: Uses small batches of data

**Optimization Algorithms**:
- SGD with Momentum
- RMSprop
- Adam (widely used)

### Backpropagation Algorithm

**Key Idea**:
- Uses output error to update previous layers' weights efficiently
- Computes gradients using the Chain Rule
- Propagates errors from output layer to input layer

---

## 4. Types of Deep Learning Models

### CNN (Convolutional Neural Networks)

<br/>

<img src="https://github.com/user-attachments/assets/e0d9dca5-cff3-4ac4-82b0-3f77b0649347" width="1000">

<br/><br/>

**Components**:
- **Convolutional Layer**: Extracts features
- **Pooling Layer**: Downsamples features
- **Fully Connected Layer**: Performs classification

### RNN (Recurrent Neural Networks)

<br/>

<img src="https://github.com/user-attachments/assets/7dd465b0-4c99-4b39-bc83-c105d133416d" width="1000">

<br/><br/>

**Characteristics**:
- Suitable for sequential data
- Maintains internal states for temporal information storage
- Uses shared parameters over time steps

**Limitations**:
- Long-term dependency issue
- Vanishing/exploding gradients

**Improved RNNs**:
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

### Transformer Architecture

<br/>

<img src="https://github.com/user-attachments/assets/29820b09-7e12-4237-9c98-519cd088abf7" width="800">

<br/><br/>

**Components**:
- **Multi-head Self-Attention**
- **Positional Encoding**
- **Encoder-Decoder Structure**

---

## 5. Practice - Simple Image Classification

### Data Preparation and Preprocessing

**Loading and Preprocessing the MNIST Dataset**:
```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data (scale to 0-1 range)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Dimension transformation (for CNN input)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Label one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Check data
print(f"Training data shape: {x_train.shape}")
print(f"Testing data shape: {x_test.shape}")
```

### Model Implementation and Training

**CNN Model Implementation**:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Model configuration
model = Sequential([
   # First convolutional layer
   Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
   MaxPooling2D(pool_size=(2, 2)),

   # Second convolutional layer
   Conv2D(64, kernel_size=(3, 3), activation='relu'),
   MaxPooling2D(pool_size=(2, 2)),

   # Convert feature maps to 1D vectors
   Flatten(),

   # Fully-connected layer
   Dense(128, activation='relu'),
   Dropout(0.5), # Prevent overfitting

   # Output layer (10 classes)
   Dense(10, activation='softmax')
])

# Model summary
model.summary()

# Model Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**Model Learning**:
```python
# Model Training
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1,
    verbose=1
)

# Visualizing the learning process
plt.figure(figsize=(12, 4))

# Accuracy graph
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss graph
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

### Evaluation and Prediction

**Model Evaluation**:
```python
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")
```

**Predictions and Visualization**:
```python
# Make predictions
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Visualize some prediction results
plt.figure(figsize=(12, 8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    
    if predicted_classes[i] == true_classes[i]:
        color = 'green'
    else:
        color = 'red'
    
    plt.title(f"예측: {predicted_classes[i]}, 실제: {true_classes[i]}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()

# Visualizing confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```

---

## Considerations for Deep Learning Training

### Overfitting Prevention Techniques
- Data Augmentation
- Dropout
- Batch Normalization
- Early Stopping
- L1/L2 Regularization

### Hyperparameter Tuning
- Learning Rate
- Batch Size
- Number of Hidden Layers and Number of Neurons
- Activation Function Selection
- Optimization Algorithm Selection

### Considerations for Practical Applications
- Balance between Model Complexity and Data Size
- Computing Resource Requirements
- Tradeoffs between Inference Time and Accuracy
- Model Interpretability
- Continuous Monitoring and Updates

---

## References and Recommended Reading

### Online Courses
- Coursera: Deep Learning Specialization (Andrew Ng)
- Fast.ai: Practical Deep Learning for Coders
- Stanford CS231n: Convolutional Neural Networks for Visual Recognition

### books
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, Aaron Courville
- "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron
- "Deep Learning with Python" by François Chollet

### Frameworks and Libraries
- TensorFlow: https://www.tensorflow.org/
- PyTorch: https://pytorch.org/
- Keras: https://keras.io/
