# Assignment 01: Training a Simple Two-Layer Neural Network with NumPy

**Related to**: Week 1 - Deep Learning Introduction

---

## A. Practicing NumPy

### A.1 Matrix Multiplication (Resource Allocation)

Allocate resources to different teams based on their performance. Use matrix multiplication to compute the total resource distribution.

- `Resources_Matrix`: 3×2 (teams × months)
- `Allocation_Factors`: 2×2 (adjustment factors)
- **Deliverable**: Compute and report the resulting 3×2 matrix.

### A.2 Element-wise Operations (Production Tracking)

Track production of three factories across seven days. Add daily production from both shifts.

- **Deliverable**: Given `Shift_A_Production` and `Shift_B_Production`, compute `Total_Production` using NumPy.

### A.3 Activation Function (Sigmoid in Sales Forecasting)

Implement the sigmoid activation function using NumPy for given forecast values.

- **Deliverable**: Function that takes an array and returns sigmoid output. Verify with the provided test values.

### A.4 Gradient Calculation (Learning Adjustment)

Implement a function to compute the gradient of the sigmoid for given inputs.

- **Deliverable**: Function that returns sigmoid gradient values for the given inputs.

---

## B. Paper ✏️

Perform **forward propagation AND backpropagation by hand** using the first data point `[20, 3, 4]` (actual output: 18).

**Given initial weights and biases**:
- **W1** (3×3): `[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]`
- **b1** (3×1): `[[0.1], [0.2], [0.3]]`
- **W2** (1×3): `[[0.2, 0.4, 0.6]]`
- **b2** (1×1): `[[0.5]]`

**Step-by-step calculations required**:

**Forward Propagation**:
1. Z1 = W1 × X + b1
2. A1 = ReLU(Z1) where ReLU(x) = max(0, x)
3. Z2 = W2 × A1 + b2
4. A2 = Sigmoid(Z2) where σ(x) = 1/(1 + e^(-x))
5. Calculate Loss = (A2 - Y)² where Y = 18

**Backward Propagation**:
1. dL/dA2 = 2(A2 - Y)
2. dL/dZ2 = dL/dA2 × sigmoid'(Z2) where sigmoid'(z) = σ(z)(1 - σ(z))
3. dL/dW2 = dL/dZ2 × A1^T
4. dL/db2 = dL/dZ2
5. dL/dA1 = W2^T × dL/dZ2
6. dL/dZ1 = dL/dA1 × ReLU'(Z1) where ReLU'(x) = 1 if x > 0, else 0
7. dL/dW1 = dL/dZ1 × X^T
8. dL/db1 = dL/dZ1

**Deliverable**: Submit your handwritten calculations (scan/photo) or typed document showing ALL intermediate values.

---

## C. NumPy 💻

**IMPORTANT**: Complete **A**, **B**, and **C** as three separate parts. B (paper) and C (code) are separate deliverables.

### Dataset

| Units Sold | Marketing Spend ($k) | Customer Satisfaction (1–5) | Total Revenue ($k) |
|------------|---------------------|-----------------------------|---------------------|
| 20         | 3                   | 4                           | 18                  |
| 15         | 5                   | 3                           | 20                  |
| 30         | 2                   | 2                           | 22                  |
| 25         | 4                   | 1                           | 25                  |
| 35         | 2                   | 3                           | 30                  |

### Architecture
- Input layer: 3 neurons
- Hidden layer: 3 neurons (**ReLU activation**)
- Output layer: 1 neuron (Sigmoid)

**Requirements**:
- Train on the complete dataset (all 5 data points)
- Implement forward propagation, backpropagation, and gradient descent
- Use a learning rate of your choice (experiment to find good value)

**Deliverable**: 
- Python code (Jupyter notebook or .py file)
- Final learned weights (W1, W2) and biases (b1, b2)
- Predictions for all 5 input data points
- Training loss over iterations (optional: plot)

---

## Submission

- Push your code (e.g. Jupyter notebook or Python script) to this repository.
- Submit **A** (code for mini-tasks), **B** (paper calculations scan/photo or document), and **C** (neural network code).
- Ensure all parts run and results are clearly shown.
