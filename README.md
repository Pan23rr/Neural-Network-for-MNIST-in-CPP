# 🧠 Neural Network From Scratch in C++

This project is a simple yet complete implementation of a Feedforward Neural Network written entirely in **C++** with no external ML libraries.

🛠️ **Goal:** Learn the inner workings of neural networks by building every component — forward propagation, backpropagation, weight updates, and evaluation — from the ground up.

---

## 🚀 Features

- Loads and processes the MNIST dataset
- Implements:
  - Sigmoid and Softmax activations
  - Categorical cross-entropy loss
  - Batch training
  - Gradient descent with learning rate tuning
- Custom training and testing loops
- Accuracy and loss tracking per batch
- Built-in benchmarking system

---

## 📊 Comparison with TensorFlow

| Feature                       | C++ Implementation     | TensorFlow        |
|-------------------------------|------------------------|-------------------|
| Library Dependency            | None                   | TensorFlow        |
| Training Speed                | Slower (manual)        | Optimized         |
| Control over internals        | Full                   | Abstracted        |
| GPU Support                   | ❌                     | ✅               |

### 📈 Accuracy Curve Comparison

**C++ Neural Network Accuracy over Batches:**

![C++ Accuracy](images/cpp_accuracy.png)

**TensorFlow Model Accuracy over 10 Epochs:**

![TensorFlow Accuracy](images/tf_accuracy.png)

---

## ⚙️ How to Build and Run

```bash
# Clone the repository
git clone https://github.com/yourusername/nn-cpp-from-scratch.git
cd nn-cpp-from-scratch

# Compile (Linux example)
g++ -O2 -std=c++17 train.cpp -o train

# Run training
./train





#Sample Output

![Sample Output](images/result.png)